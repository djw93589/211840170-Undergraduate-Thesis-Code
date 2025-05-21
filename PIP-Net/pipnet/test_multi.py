from tqdm import tqdm
import numpy as np
import torch, torch.nn.functional as F
from util.log import Log
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

@torch.no_grad()
def eval_pipnet(
        net,
        test_loader,
        epoch,
        device,
        log: Log = None,
        threshold: float = 0.1,      # 逐类判正阈值
        progress_prefix: str = 'Eval Epoch'
    ):

    net = net.to(device)
    net.eval()

    K = net.module._num_classes
    tp = torch.zeros(K, device=device)
    fp = torch.zeros(K, device=device)
    fn = torch.zeros(K, device=device)

    global_sim_anz   = 0.
    global_anz       = 0.
    local_size_total = 0.
    abstained        = 0

    all_probs  = []
    all_labels = []

    test_iter = tqdm(enumerate(test_loader), total=len(test_loader),
                     desc=f'{progress_prefix} {epoch}', ncols=0, mininterval=5.)

    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device).float()   # ys 多热

        _, pooled, out = net(xs, inference=False)

        if i == 0:
            print("\n[DEBUG Batch 0]")
            print("ys.shape:", ys.shape)
            print("ys[0]:", ys[0].cpu().numpy())
            print("pooled[0]:", pooled[0].detach().cpu().numpy())
            print("pooled > 1e-3:", (pooled[0] > 1e-3).sum().item(), "non-zero dimensions")
            print("weight > 1e-3:", (net.module._classification.weight > 1e-3).sum(dim=1).cpu().numpy())

        logits = out
        probs = torch.sigmoid(logits)
        y_hat = (probs >= 0.575).float()
        
        if i == 0:
            print("\n[DEBUG Batch 0]")
            print("ys.shape:", ys.shape)
            print("ys[0]:", ys[0].cpu().numpy())
            print("out[0]:", out[0].detach().cpu().numpy())
            print("probs[0]:", probs[0].detach().cpu().numpy())
            print("y_hat[0]:", y_hat[0].detach().cpu().numpy())
            print("probs.max():", probs.max().item())
        
        max_prob, _ = probs.max(dim=1)
        abstained  += (max_prob < threshold/10).sum().item()

        # ── TP / FP / FN 统计 ─────────────────────────────────────
        tp += (y_hat * ys).sum(dim=0)
        fp += (y_hat * (1 - ys)).sum(dim=0)
        fn += ((1 - y_hat) * ys).sum(dim=0)

        # ── 原型 & 局部解释统计（保持原逻辑）──────────────────────
        rep_w = net.module._classification.weight.unsqueeze(1).repeat(1, pooled.shape[0], 1)
        sim_scores_anz = torch.count_nonzero(torch.gt(torch.abs(pooled * rep_w), 1e-3).float(), dim=2).float()
        local_size = torch.count_nonzero(torch.gt(torch.relu((pooled * rep_w) - 1e-3).sum(dim=1), 0.).float(), dim=1).float()

        correct_sim_anz = torch.diagonal(sim_scores_anz)  # 多标签下约等处理
        global_sim_anz += correct_sim_anz.sum().item()
        almost_nz = torch.count_nonzero(torch.gt(torch.abs(pooled), 1e-3).float(), dim=1).float()
        global_anz += almost_nz.sum().item()
        local_size_total += local_size.sum().item()

        all_probs.append(probs.detach().cpu())
        all_labels.append(ys.detach().cpu())

    # ───────────────────────── 计算指标 ─────────────────────────
    eps = 1e-9
    macro_f1  = (2 * tp / (2 * tp + fp + fn + eps)).mean().item()
    micro_f1  = (2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + eps)).item()

    y_true = torch.vstack(all_labels).numpy()
    y_scr  = torch.vstack(all_probs).numpy()
    valid_mask = np.logical_and(y_true.sum(0) > 0, (1-y_true).sum(0) > 0)
    y_true   = y_true[:, valid_mask]
    y_scr    = y_scr[:,  valid_mask]
    mAP    = average_precision_score(y_true, y_scr, average='macro')
    auc    = roc_auc_score(y_true, y_scr, average='macro')
    
    info = dict(
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        mAP=mAP,
        macro_auc=auc,
        almost_sim_nonzeros = global_sim_anz / len(test_loader.dataset),
        local_size_all_classes = local_size_total / len(test_loader.dataset),
        almost_nonzeros = global_anz / len(test_loader.dataset),
        sparsity = (torch.numel(net.module._classification.weight) -
                    torch.count_nonzero(torch.relu(net.module._classification.weight - 1e-3)).item())
                   / torch.numel(net.module._classification.weight),
        abstained = abstained,
    )

    print(f"\n[E{epoch}] macro‑F1: {macro_f1:.4f} | micro‑F1: {micro_f1:.4f} "
          f"| mAP: {mAP:.4f} | macro‑AUC: {auc:.4f} | abstained: {abstained}")

    return info
