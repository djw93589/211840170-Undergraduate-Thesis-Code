import argparse, os, random, time
from pathlib import Path
from typing import Tuple, List

import torch, torchvision
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision.models import (
    resnet18, ResNet18_Weights,
    densenet121, DenseNet121_Weights
)
from PIL import Image

class KneeCUBStyle(Dataset):
    def __init__(self, root:Path, split:str,
                 transform=None, use_bbox=False,
                 val_ratio=0.1, seed=42):

        self.transform = transform
        self.use_bbox = use_bbox
        # 读 4 份文件
        imgs_f = (root/'images.txt').read_text().strip().splitlines()
        split_f = (root/'train_test_split.txt').read_text().strip().splitlines()
        bbox_f = (root/'bounding_boxes.txt').read_text().strip().splitlines()
        class_f = (root/'classes.txt').read_text().strip().splitlines()

        # id → path
        id2path = {
        int(l.split(maxsplit=1)[0]): l.split(maxsplit=1)[1]
            for l in imgs_f
            }
        # id → split flag
        id2flag = {int(l.split()[0]): int(l.split()[1]) for l in split_f}  # 1=train ,0=test
        # id → bbox
        id2bbox = {int(l.split()[0]): tuple(map(float,l.split()[1:])) for l in bbox_f}
        # path → label idx
        cls2idx = {line.split()[1]: int(line.split()[0])-1 for line in class_f}

        ids = sorted(id2path.keys())
        # train pool   = flag==1
        # test  set    = flag==0
        train_ids = [i for i in ids if id2flag[i]==1]
        test_ids  = [i for i in ids if id2flag[i]==0]

        # 从 train_ids 再随机抽 val_ratio 作为 val
        random.seed(seed)
        random.shuffle(train_ids)
        val_len = int(len(train_ids)*val_ratio)
        val_ids = train_ids[:val_len]
        train_ids = train_ids[val_len:]

        if split=='train': use_ids = train_ids
        elif split=='val': use_ids = val_ids
        else:              use_ids = test_ids

        self.samples: List[Tuple[str,int,int]] = []   # (abs_path, label, id)
        for idx in use_ids:
            rel_path = id2path[idx]
            abs_path = root/'images'/rel_path
            label = cls2idx[rel_path.split('/')[0]]
            self.samples.append((abs_path, label, idx))

        self.id2bbox = id2bbox
        self.split = split

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, label, idx = self.samples[i]
        img = Image.open(path).convert('L')  # 灰度
        if self.use_bbox:
            x,y,w,h = self.id2bbox[idx]
            img = img.crop((x,y,x+w,y+h))
        if self.transform: img = self.transform(img)
        return img, label

# --------------------------- 构建 DataLoader -------------------------------- #
def build_loaders(root:Path, batch:int, workers:int,
                  use_bbox=False, val_ratio=0.1):
    tfm = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.5],[0.5])
    ])
    train_ds = KneeCUBStyle(root,'train',tfm,use_bbox,val_ratio)
    val_ds   = KneeCUBStyle(root,'val',  tfm,use_bbox,val_ratio)
    test_ds  = KneeCUBStyle(root,'test', tfm,use_bbox,val_ratio)

    train_ld = DataLoader(train_ds,batch_size=batch,shuffle=True,
                          num_workers=workers,pin_memory=True)
    val_ld   = DataLoader(val_ds,batch_size=batch,shuffle=False,
                          num_workers=workers,pin_memory=True)
    test_ld  = DataLoader(test_ds,batch_size=batch,shuffle=False,
                          num_workers=workers,pin_memory=True)
    return train_ld,val_ld,test_ld,5  # num_classes=5

# --------------------------- 模型 ------------------------------------------ #
def build_model(arch:str, num_classes:int)->nn.Module:
    if arch=='resnet18':
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        m.fc = nn.Linear(m.fc.in_features,num_classes)
    elif arch=='densenet121':
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        m.features.conv0 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        m.classifier = nn.Linear(m.classifier.in_features,num_classes)
    else:
        raise ValueError("arch 必须是 resnet18 / densenet121")
    return m

# --------------------------- 训练 / 验证 ------------------------------------ #
def run_epoch(model,loader,criterion,optimizer,device,train=True):
    model.train() if train else model.eval()
    loss_sum,correct,total = 0.0,0,0
    torch.set_grad_enabled(train)
    for x,y in loader:
        x,y = x.to(device),y.to(device)
        out = model(x)
        loss = criterion(out,y)
        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_sum += loss.item()
        pred = out.argmax(1)
        total += y.size(0); correct += pred.eq(y).sum().item()
    return loss_sum/len(loader), 100.*correct/total

# --------------------------- 主函数 ---------------------------------------- #
def main(args):
    random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = Path(args.data_dir).expanduser()
    tr_ld,val_ld,te_ld,num_classes = build_loaders(
        root,args.batch_size,args.num_workers,
        use_bbox=args.use_bbox,val_ratio=args.val_ratio)

    model = build_model(args.arch,num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=args.epochs,eta_min=args.lr*0.01)

    best=0.0; save_p = Path(args.save_path).expanduser()
    save_p.parent.mkdir(exist_ok=True,parents=True)

    for ep in range(1,args.epochs+1):
        t0=time.time()
        tr_loss,tr_acc = run_epoch(model,tr_ld,criterion,optimizer,device,train=True)
        val_loss,val_acc= run_epoch(model,val_ld,criterion,optimizer,device,train=False)
        scheduler.step()
        print(f"[{ep:02d}/{args.epochs}] "
              f"train {tr_loss:.3f}/{tr_acc:.2f}% | "
              f"val {val_loss:.3f}/{val_acc:.2f}%  "
              f"{time.time()-t0:.1f}s")
        if val_acc>best:
            best=val_acc; torch.save(model.state_dict(),save_p)
            print(f"  Save best ({best:.2f}%)")

    # -------- 测试 --------
    model.load_state_dict(torch.load(save_p,map_location=device))
    te_loss,te_acc = run_epoch(model,te_ld,criterion,optimizer,device,train=False)
    print(f" Test Accuracy: {te_acc:.2f}%  (loss {te_loss:.3f})")
    print(f"Model saved at {save_p}")

# --------------------------- CLI ------------------------------------------- #
if __name__=="__main__":
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_dir", required=True,
                    help="指向 MedicalExpert-1 根目录（包含 images.txt 等）")
    ap.add_argument("--arch", choices=["resnet18","densenet121"],
                    default="densenet121")
    ap.add_argument("--use_bbox", action="store_true",
                    help="使用 bounding_boxes.txt 对 ROI 进行裁剪")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="从训练 IDs 中随机划分验证集比例")
    ap.add_argument("--save_path", default="./checkpoint/knee_best.pth")
    args = ap.parse_args()
    main(args)
