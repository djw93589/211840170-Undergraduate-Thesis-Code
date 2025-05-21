import os
import torch
from torchvision.utils import save_image
from medmnist import PathMNIST
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def safe_image_to_tensor(img):
    # """安全地将 MedMNIST 图像转为 PyTorch 张量"""
    if not isinstance(img, np.ndarray):
        img = np.array(img)  # 强制转 NumPy
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)  # 确保是 uint8
    return torch.tensor(img).permute(2, 0, 1).float() / 255.

save_root = '../../data/PathMNIST/images'

# 1. 加载原始数据集
train_dataset = PathMNIST(split="train", download=True)
test_dataset = PathMNIST(split="test", download=True)

def stratified_sample(data, sample_frac=0.1):
    """对数据集进行分层抽样"""
    labels = data.labels
    # 多标签任务：选择第一个正类作为分层依据（简化处理）
    stratify_by = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    # 分层抽样
    _, subset_indices = train_test_split(
        range(len(labels)),
        test_size=sample_frac,
        stratify=stratify_by,
        random_state=42  # 固定随机种子保证可复现
    )

    # 返回抽样后的数据
    return Subset(data, subset_indices)
    
# 2. 对每个划分抽样1/10
sampled_train = stratified_sample(train_dataset, sample_frac=0.1)
sampled_test = stratified_sample(test_dataset, sample_frac=0.1)
dataset = sampled_train + sampled_test

label_to_name = {i: f"{i:03d}.Class{i}" for i in range(len(dataset))}

images_txt = []
labels_txt = []
split_txt = []
bbox_txt = []

for i in tqdm(range(len(dataset))):
    img, label = dataset[i]
    label = int(label)
    class_name = label_to_name[label]
    
    save_dir = os.path.join(save_root, class_name)
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{class_name}_{i:04d}.png"
    save_path = os.path.join(save_dir, filename)
    img = safe_image_to_tensor(img)
    save_image(img, save_path)

    full_path = os.path.relpath(save_path, start='../../data/PathMNIST/images').replace('\\', '/')
    images_txt.append(f"{i+1} {full_path}")
    labels_txt.append(f"{i+1} {label+1}")
    split_txt.append(f"{i+1} {1 if i < len(dataset) * 0.8 else 0}")
    bbox_txt.append(f"{i+1} 0 0 28 28")  # 原图是28×28，默认全图bbox

    # 写入CUB风格的 txt 文件
    with open('../../data/PathMNIST/images.txt', 'w') as f:
        f.write('\n'.join(images_txt))
    with open('../../data/PathMNIST/image_class_labels.txt', 'w') as f:
        f.write('\n'.join(labels_txt))
    with open('../../data/PathMNIST/train_test_split.txt', 'w') as f:
        f.write('\n'.join(split_txt))
    with open('../../data/PathMNIST/bounding_boxes.txt', 'w') as f:
        f.write('\n'.join(bbox_txt))
    with open('../../data/PathMNIST/classes.txt', 'w') as f:
        for i in range(len(dataset)):
            f.write(f"{i+1} {label_to_name[i]}\n")