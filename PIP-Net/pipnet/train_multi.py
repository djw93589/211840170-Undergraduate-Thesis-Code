from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import math

def debug_classifier_state(net, out, ys, epoch, batch_idx):
    classifier = net.module._classification

    # 打印分类层是否可训练
    print(f"[DEBUG] Epoch {epoch} Batch {batch_idx}")
    print(f"  requires_grad: {classifier.weight.requires_grad}")
    
    # 打印分类层的权重均值和标准差
    print(f"  weight mean: {classifier.weight.mean().item():.6f}, std: {classifier.weight.std().item():.6f}")

    # 检查分类层的梯度（需在 loss.backward() 之后调用）
    if classifier.weight.grad is not None:
        print(f"  grad mean: {classifier.weight.grad.mean().item():.6f}, std: {classifier.weight.grad.std().item():.6f}")
    else:
        print("  grad: None (possible loss not propagated)")

    # 检查输出预测和标签
    probs = torch.sigmoid(out)
    y_hat = (probs > 0.5).float()

    print("  probs[0]:", probs[0].detach().cpu().numpy())
    print("  y_hat[0]:", y_hat[0].cpu().numpy())
    print("  ys[0]:   ", ys[0].cpu().numpy())

    if torch.all(y_hat == 1) or torch.all(y_hat == 0):
        print("   Warning: Model predicts all 1 or all 0 — may not be learning.")


def train_pipnet(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion,
                 epoch, nr_epochs, device, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch'):
    # Make sure the model is in train mode
    net.train()

    if pretrain:
        # Disable training of classification layer
        for param in net.module._classification.parameters():
            param.requires_grad = True
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        for param in net.module._classification.parameters():
            param.requires_grad = True

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    iters = len(train_loader)
    # Show progress on progress bar.
    train_iter = tqdm(enumerate(train_loader),
                      total=len(train_loader),
                      desc=progress_prefix + '%s' % epoch,
                      mininterval=2.,
                      ncols=0)

    count_param = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param += 1
    print("Number of parameters that require gradient: ", count_param, flush=True)

    # 动态权重设置
    if pretrain:
        align_pf_weight = (epoch / nr_epochs) * 1.
        unif_weight = 0.5  # ignored
        t_weight = 5.
        cl_weight = 2.
    else:
        align_pf_weight = 5.
        t_weight = 0.5
        unif_weight = 0.
        cl_weight = 2.
        
    print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, "Class weight:", cl_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)
    
    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:  # y_s是标签

        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)

        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)

        # Perform a forward pass through the network
        proto_features, pooled, out = net(torch.cat([xs1, xs2]))
        loss, acc = calculate_loss(proto_features, pooled, out, ys, align_pf_weight, t_weight, unif_weight, cl_weight,
                                   net.module._classification.normalization_multiplier, pretrain, finetune, criterion,
                                   train_iter, print=True, EPS=1e-8)
        
        # DEBUG POINT 1: check if classification layer is trainable
        if i == 0 and epoch == 1:
            print("[DEBUG] Classification weight requires_grad:",
                  net.module._classification.weight.requires_grad, flush=True)
            print("[DEBUG] Mean weight value before training:",
                  torch.mean(net.module._classification.weight).item(), flush=True)
            print("[DEBUG] Classification weight std:", net.module._classification.weight.std().item())
            print("[DEBUG] Raw out stats: max={:.4f}, min={:.4f}, mean={:.4f}".format(
                    out.max().item(), out.min().item(), out.mean().item()))
            
        # Compute the gradient
        loss.backward()
        
        if not pretrain:
            optimizer_classifier.step()
            scheduler_classifier.step(epoch - 1 + (i / iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])

        if not finetune:
            optimizer_net.step()
            scheduler_net.step()
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.)

        with torch.no_grad():
            total_acc += acc
            total_loss += loss.item()

        # DEBUG POINT 2: check if logits/probs/y_hat are reasonable
        if i == 0:
            logits = out[:1]
            probs = torch.sigmoid(logits)
            y_hat = (probs >= 0.575).float()
            
            print("[DEBUG] Multiplier:", net.module._classification.normalization_multiplier.item())
            print("[DEBUG] Sample ys[0]:", ys[:1].cpu().numpy())
            print("[DEBUG] Raw out[0]:", out[:1].detach().cpu().numpy())
            print("[DEBUG] Logits[0]:", logits.detach().cpu().numpy())
            print("[DEBUG] Sigmoid probs[0]:", probs.detach().cpu().numpy())
            print("[DEBUG] y_hat[0]:", y_hat.cpu().numpy())

        # DEBUG POINT 3: weight after optimizer step
        if i == 0 and epoch == 1:
            with torch.no_grad():
                num_nonzero = torch.count_nonzero(net.module._classification.weight).item()
                mean_weight = torch.mean(net.module._classification.weight).item()
                print(f"[DEBUG] After backward+step: mean weight: {mean_weight:.6f}, non-zero count: {num_nonzero}", flush=True)

    train_info['train_accuracy'] = total_acc / float(i + 1)
    train_info['loss'] = total_loss / float(i + 1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class

    return train_info
    
def calculate_loss(proto_features, pooled, out, ys1,
                   align_pf_weight, t_weight, unif_weight, cl_weight,
                   net_normalization_multiplier, pretrain, finetune,
                   criterion, train_iter, print=True, EPS=1e-10):

    ys = torch.cat([ys1, ys1])
    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)

    embv1 = pf1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
    embv2 = pf2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)

    # --- 原型对齐损失 ---
    a_loss_pf = (align_loss(embv1, embv2.detach()) + align_loss(embv2, embv1.detach())) / 2.

    # --- tanh 激活正则 ---
    tanh_loss = -(torch.log(torch.tanh(torch.sum(pooled1, dim=0)) + EPS).mean() +
                  torch.log(torch.tanh(torch.sum(pooled2, dim=0)) + EPS).mean()) / 2.

    # --- 总 loss 初始化 ---
    loss = align_pf_weight * a_loss_pf + t_weight * tanh_loss

    # --- 始终加分类损失 ---
    logits = out
    probs = torch.sigmoid(logits)
    class_loss = criterion(logits, ys.float())
    loss = loss + cl_weight * class_loss

    # --- 计算准确率 ---
    acc = 0.
    y_hat = (probs > 0.575).float()
    correct = (y_hat == ys).float()
    acc = correct.sum(dim=1).div(ys.size(1)).mean().item()

    # --- 打印 ---
    if print:
        with torch.no_grad():
            postfix = f"L:{loss.item():.3f}, LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled - 0.1), dim=1).float().mean().item():.1f}, Ac:{acc:.3f}"
            train_iter.set_postfix_str(postfix, refresh=False)

    return loss, acc



# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. Currently not used but you could try adding it if you want.
def uniform_loss(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss


# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False

    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss