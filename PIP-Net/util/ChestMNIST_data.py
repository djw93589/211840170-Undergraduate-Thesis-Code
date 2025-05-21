import os
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# --------------------------------------------------
# 1. ChestMNISTCUBStyle — load CUB‑style txt & images
# --------------------------------------------------
class ChestMNISTCUBStyle(Dataset):
    def __init__(self, root: str | Path, split: str = 'train', *, transform=None, double_view: bool = False):
        self.base_ds = self
        self.root = Path(root)
        self.double_view = double_view
        self.transform = transform  # None ⇒ leave as PIL, caller decides

        # -- images.txt -------------------------------------------------------
        with open(self.root / 'images.txt') as f:
            lines = [ln.strip().split() for ln in f]
        self.image_paths = [self.root / 'images' / path for _, path in lines]

        # -- labels -----------------------------------------------------------
        with open(self.root / 'image_class_labels.txt') as f:
            lab_lines = [ln.strip().split(maxsplit=1) for ln in f]
        self.labels: List[torch.Tensor] = []
        for _id, lab in lab_lines:
            vec = torch.zeros(14, dtype=torch.float32)
            lab = lab.translate({91: None, 93: None}).replace(',', ' ').replace('_', ' ').strip()
            if lab.lower() != 'none':
                for i in map(int, lab.split()):
                    vec[i] = 1.
            self.labels.append(vec)

        # -- train / test split ----------------------------------------------
        with open(self.root / 'train_test_split.txt') as f:
            flags = [int(ln.strip().split()[1]) for ln in f]
        idx_all = list(range(len(flags)))
        self.indices = [i for i in idx_all if (flags[i] == 1 and split == 'train') or (flags[i] == 0 and split != 'train')]
        self.class_to_idx = {str(i): i for i in range(14)}
        self.imgs = list(zip(self.image_paths, self.labels))
        self.samples = list(zip(self.image_paths, self.labels))
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_pil  = Image.open(self.image_paths[real_idx]).convert('RGB')
        label    = self.labels[real_idx]

        img_proc = self.transform(img_pil) if self.transform else img_pil

        if self.double_view:
            # 返回两份相同视图，让上层自行再做增强
            return img_proc, img_proc, label
        return img_proc, label

# --------------------------------------------------
# 2. Two‑view Dataset wrapper (SimCLR‑style)
# --------------------------------------------------
class TwoAugSupervisedDataset(Dataset):
    def __init__(self, base_ds: Dataset, transform1, transform2):
        self.base_ds, self.t1, self.t2 = base_ds, transform1, transform2
        
    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        # 如果上层误传 Tensor，这里兜底转回 PIL
        if torch.is_tensor(img):
            img = T.ToPILImage()(img)
        return self.t1(img), self.t2(img), label

# --------------------------------------------------
# 3. Transform helpers
# --------------------------------------------------

def get_transforms(img_size=224, augment=True):
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if not augment:
        base = [T.Resize((img_size, img_size)), T.ToTensor(), normalize]
        return T.Compose(base), T.Compose(base)

    aug1 = T.Compose([
        T.Resize((img_size + 8, img_size + 8)),
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])
    aug2 = T.Compose([
        T.Resize((img_size + 8, img_size + 8)),
        T.ColorJitter(0.4, 0.4, 0.4),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        T.RandomCrop(img_size),
        T.ToTensor(),
        normalize
    ])
    return aug1, aug2

# --------------------------------------------------
# 4. get_data  (returns datasets)
# --------------------------------------------------

def get_data(args):
    t1, t2       = get_transforms(args.image_size, augment=True)
    t_no_aug, _  = get_transforms(args.image_size, augment=False)

    # base (no transform) ⇒ returns PIL
    base_train = ChestMNISTCUBStyle(args.data_path, split='train', transform=None)

    train_ds   = TwoAugSupervisedDataset(base_train, t1, t2)
    train_norm = ChestMNISTCUBStyle(args.data_path, split='train', transform=t_no_aug)
    train_aug  = ChestMNISTCUBStyle(args.data_path, split='train', transform=T.Compose([t1, t2]))
    test_ds    = ChestMNISTCUBStyle(args.data_path, split='test',  transform=t_no_aug)
    proj_ds    = ChestMNISTCUBStyle(args.data_path, split='train', transform=t_no_aug)

    num_classes = 14
    return (train_ds, train_ds, train_norm, train_aug,
            proj_ds, test_ds, proj_ds,
            list(range(num_classes)), 3,
            list(range(len(base_train))), torch.zeros(len(base_train), dtype=torch.long))

# --------------------------------------------------
# 5. get_dataloaders
# --------------------------------------------------

def get_dataloaders(args, device):
    (train_set, train_set_p, train_norm, train_aug,
     proj_set, test_set, test_proj, classes, num_ch, train_idx, targets) = get_data(args)

    def mk(dl_set, bs, shuf, drop):
        return DataLoader(dl_set, bs, shuffle=shuf, drop_last=drop,
                          pin_memory=True, num_workers=args.num_workers,
                          worker_init_fn=lambda _: np.random.seed(args.seed))

    train_loader              = mk(train_set,       args.batch_size,         True,  True)
    train_loader_pretraining  = mk(train_set_p,     args.batch_size_pretrain, True,  True)
    train_loader_normal       = mk(train_norm,      args.batch_size,         True,  True)
    train_loader_norm_aug     = mk(train_aug,       args.batch_size,         True,  True)
    project_loader            = mk(proj_set,        1,                       False, False)
    test_loader               = mk(test_set,        args.batch_size,         False, False)
    test_project_loader       = mk(test_proj,       1,                       False, False)

    return (train_loader, train_loader_pretraining, train_loader_normal,
            train_loader_norm_aug, project_loader, test_loader,
            test_project_loader, classes)
