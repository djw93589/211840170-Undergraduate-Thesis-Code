# 项目简介
本项目从可解释性的角度出发，围绕 PIP-Net 模型对多种医学图像分类任务进行了研究与实验。通过在 MedMNIST、膝盖关节炎 X-ray 与儿童肺炎 X-ray 三个典型数据集上的实验，验证了 PIP-Net 在不同分类设定下的适应性与可视化能力。

# 环境依赖与安装
建议使用 Conda 创建隔离环境：
```bash
conda create -n pipnet_env python=3.12
conda activate pipnet_env
pip install -r requirements.txt`
```
# 数据集下载
- 对于MedMNIST数据集，可用以下代码下载：
```bash
pip install medmnist
```
```python
import medmnist
from medmnist import RetinaMNIST
```
- 对于2017儿童肺炎 X-ray 数据集以及膝盖关节炎 X-ray 数据集 建议下载到本地。

# 训练命令
 1. CAM系列算法训练如下：
```bash
python train_KneeXray_res_dense.py \
  --data_dir ../data/KneeXray/MedicalExpert-1 \
  --arch densenet121 \
  --save_path ./checkpoint/MedicalExpert-1_dense
```
```bash
python run_cam.py \
    --arch densenet121 \
    --model_path ./checkpoint/dense_MedicalExpert-1.pth \
    --image_path "../data/KneeXray/MedicalExpert-1/images/3Moderate/01224_ModerateG3 (16).png" \
    --method eigencam \
    --output_dir ./results \
    --output "01224_ModerateG3_(16)_densenet121_eigen.png"
```
- PIP-Net训练代码如下：
```bash
python main.py 
	--dataset ../data/KneeXray/MedicalExpert-1 \
	--batch_size 32 \
	--batch_size_pretrain 64 \
	--epochs 30 \
	--epochs_pretrain 10 \
	--log_dir ./runs/run_KneeXray_1_pipnet
```
# 代码文档说明
code文件夹中分为了CAM、PIP-Net、Traditional三个子文件夹，下面逐一说明：
## CAM 文件夹
-  hash_matcher.py 利用哈希比对寻找两位专家人工标注不同的图片；
- train_KneeXray_res_dense.py 用于生成.pth 文件作为CAM系列算法载体；
- run_cam.py 用于可视化CAM系列算法。

## PIP-Net文件夹
### pipnet文件夹
- train.py 、test.py 、pipnet.py 为PIP-Net的训练、测试和模型搭建；
- train_multi.py、test_multi.py 为改进后对于多标签问题的训练和测试代码。

### util文件夹
- arg.py 包含模型的参数设置；
- data.py 包含模型的数据加载设置；
- 后缀为_data.py的文件对应数据的加载；
- 前缀为preprocess的文件对应数据的预处理；
- vis_pipnet.py和 visualize_prediction.py为可视化相关代码。
### main.py 主函数

## Traditional文件夹
- MedMNIST 数据集上的传统机器学习算法已有对应名称命名python文件。

# 致谢
本项目使用了公开医学图像数据集 [ChestMNIST](https://medmnist.com/)，由 Yang 等人在 2021 年发布，特此致谢其作者团队为医学图像社区提供的数据资源支持。

部分模型结构与训练流程参考并改编自 [PIP-Net](https://github.com/M-Nauta/PIPNet.) 的开源实现，感谢原作者的贡献与分享。

同时，感谢 PyTorch 与 torchvision 项目为深度学习研究提供了稳定而强大的框架支持。

