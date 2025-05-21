import os
import shutil
from PIL import Image

# =======================================
# 路径配置 —— 按需修改
# =======================================
original_dataset_dir = './'  # 请根据实际存放路径修改
target_dataset_dir = '../data/ChestXray3/'  # 目标数据集存放路径

# =======================================
# 创建目标目录结构
# =======================================
images_dir = os.path.join(target_dataset_dir, 'images')
class_names = ['BACTERIA', 'VIRUS', 'NORMAL']     # 3 类
for cls in class_names:
    os.makedirs(os.path.join(images_dir, cls), exist_ok=True)

# 说明文件列表
images_lines, split_lines, bbox_lines = [], [], []
split_mapping = {'train': 1, 'val': 1, 'test': 0}  # val 并入训练
image_id = 1                                       # 全局唯一 ID

def copy_and_record(src, cls, split_flag):
    """复制图像并记录到说明文件"""
    global image_id
    new_fname      = f"{image_id:05d}_{os.path.basename(src)}"
    rel_path       = os.path.join(cls, new_fname)
    dst            = os.path.join(images_dir, rel_path)
    shutil.copy2(src, dst)

    # 读取尺寸
    try:
        with Image.open(dst) as im:
            w, h = im.size
    except Exception as e:
        print(f"[读取失败] {dst}: {e}"); return

    images_lines.append(f"{image_id} {rel_path}")
    split_lines.append(f"{image_id} {split_flag}")
    bbox_lines.append(f"{image_id} 0 0 {w} {h}")
    image_id += 1

# =======================================
# 遍历 splits
# =======================================
for split_folder, split_flag in split_mapping.items():
    split_dir = os.path.join(original_dataset_dir, split_folder)
    if not os.path.isdir(split_dir):
        print(f"跳过缺失目录: {split_dir}")
        continue

    # ------ NORMAL ------
    normal_dir = os.path.join(split_dir, 'NORMAL')
    if os.path.isdir(normal_dir):
        for f in os.listdir(normal_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                copy_and_record(os.path.join(normal_dir, f), 'NORMAL', split_flag)

    # ------ PNEUMONIA -> BACTERIA / VIRUS ------
    pneu_dir = os.path.join(split_dir, 'PNEUMONIA')
    if os.path.isdir(pneu_dir):
        for f in os.listdir(pneu_dir):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            lower = f.lower()
            if 'bacteria' in lower:
                cls = 'BACTERIA'
            elif 'virus' in lower:
                cls = 'VIRUS'
            else:
                print(f"[未识别子类] {f} -> 跳过"); continue
            copy_and_record(os.path.join(pneu_dir, f), cls, split_flag)

# =======================================
# 写入说明文件
# =======================================
os.makedirs(target_dataset_dir, exist_ok=True)
with open(os.path.join(target_dataset_dir, 'images.txt'), 'w') as f:
    f.write('\n'.join(images_lines))
with open(os.path.join(target_dataset_dir, 'train_test_split.txt'), 'w') as f:
    f.write('\n'.join(split_lines))
with open(os.path.join(target_dataset_dir, 'bounding_boxes.txt'), 'w') as f:
    f.write('\n'.join(bbox_lines))
with open(os.path.join(target_dataset_dir, 'classes.txt'), 'w') as f:
    f.write('\n'.join(f"{i+1} {c}" for i, c in enumerate(class_names)))

print(f"数据集重构完成！共处理 {image_id-1} 张图像，存放于: {target_dataset_dir}")

