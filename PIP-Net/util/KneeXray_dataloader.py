import os
import shutil
from PIL import Image
import random

original_dataset_dir = './'  # 请根据实际存放路径修改
target_dataset_dir = '../../data/KneeXray/MedicalExpert-1/'  # 目标数据集存放路径

# ============================
# 创建目标数据集目录结构
# ============================
images_dir = os.path.join(target_dataset_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

class_names = ['0Normal', '1Doubtful', '2Mild', '3Moderate', '4Severe']
for cls in class_names:
    cls_dir = os.path.join(images_dir, cls)
    os.makedirs(cls_dir, exist_ok=True)

# ============================
# 定义用于生成说明文件的列表
# ============================
images_lines = []  # 存储 images.txt 中的内容，每一行格式： image_id relative_path
split_lines = []   # 存储 train_test_split.txt 中的内容，每一行格式： image_id split（1表示训练，0表示测试）
bbox_lines = []

image_id = 1  # 用于标记每个图像的唯一 id

# ============================
# 遍历各个数据分割文件夹，将图像复制到新的结构中
# ============================
for cls in class_names:
    cls_source_dir = os.path.join(original_dataset_dir, cls)
    if not os.path.exists(cls_source_dir):
        print(f"目录 {cls_source_dir} 不存在，跳过。")
        continue

        # 遍历当前类别目录下所有图像文件（根据文件后缀过滤）
    for fname in os.listdir(cls_source_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            source_path = os.path.join(cls_source_dir, fname)
            # 为了保证新文件名唯一，可在原文件名前添加 image_id 作为前缀
            new_fname = f"{image_id:05d}_{fname}"
            # 新的相对路径：类别子文件夹 + 文件名（与 CUB 数据集类似相对路径）
            relative_path = os.path.join(cls, new_fname)
            target_path = os.path.join(images_dir, relative_path)
            output = random.choices([1, 0], weights=[0.8, 0.2], k=1)[0]

            # 复制图像到目标位置
            shutil.copy2(source_path, target_path)

            try:
                with Image.open(target_path) as img:
                    width, height = img.size  # 注意：PIL 中 size 返回 (宽, 高)
            except Exception as e:
                print(f"读取图像 {target_path} 出错：{e}")
                continue

                
            # 添加 images.txt 的记录，注意这里记录的相对路径为相对于目标数据集的根目录
            images_lines.append(f"{image_id} {relative_path}")
            # 添加 train_test_split.txt 的记录：1 表示训练(包括 train 和 val)，0 表示测试
            split_lines.append(f"{image_id} {output}")
            bbox_lines.append(f"{image_id} 0 0 {width} {height}")
                
            image_id += 1

# ============================
# 生成 classes.txt 文件
# ============================
classes_lines = []
for idx, cls in enumerate(class_names, start=1):
    classes_lines.append(f"{idx} {cls}")

# ============================
# 将说明文件写入目标数据集目录
# ============================
with open(os.path.join(target_dataset_dir, 'images.txt'), 'w') as f:
    f.write("\n".join(images_lines))

with open(os.path.join(target_dataset_dir, 'train_test_split.txt'), 'w') as f:
    f.write("\n".join(split_lines))

with open(os.path.join(target_dataset_dir, 'classes.txt'), 'w') as f:
    f.write("\n".join(classes_lines))
    
with open(os.path.join(target_dataset_dir, 'bounding_boxes.txt'), 'w') as f:
    f.write("\n".join(bbox_lines))
print("数据集重构完成！")
