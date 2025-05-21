import argparse, os
import torch
import torch.nn.functional as F
import numpy as np
import cv2, matplotlib
matplotlib.use("Agg")                      # 防止无 GUI
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pathlib import Path
from train_KneeXray_res_dense import build_model        # ← 直接复用你训练脚本里的函数

# -------------------- Grad‑CAM util -------------------- #
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.target_layers = target_layer

        self.activations = None
        self.gradients = None

        # forward hook
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        # backward hook
        def bwd_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x: torch.Tensor, class_idx=None):
        """
        x:  (1, 1, 224, 224)
        return: (H, W) heat‑map, already normalized [0,1]
        """
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # GAP over gradients → weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (C,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-9
        return cam, class_idx, score.item()

class GradCAMPlusPlus(GradCAM):
    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[0, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients  # (B, C, H, W)
        acts = self.activations  # (B, C, H, W)

        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)

        eps = 1e-8
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom) * eps)
        alphas = alpha_num / alpha_denom

        weights = (alphas * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min(); cam /= cam.max() + 1e-9
        return cam, class_idx, score.item()


class EigenCAM(GradCAM):
    def __call__(self, x, class_idx=None):
        _ = self.model(x)
        acts = self.activations[0]  # shape: (C, H, W)
        C, H, W = acts.shape

        acts_flat = acts.view(C, -1)  # (C, H*W), keep on GPU
        u, s, vh = torch.linalg.svd(acts_flat, full_matrices=False)
        weights = u[:, 0].view(-1, 1)

        cam = torch.matmul(acts_flat.T, weights).view(1, 1, H, W)  # (1,1,H,W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min(); cam /= cam.max() + 1e-9
        return cam.detach().cpu().numpy(), -1, 0.0

        
# ------------------ helper: pick target layer ----------------- #
def get_last_conv_layer(model, arch):
    if arch == "resnet18":
        return model.layer4[-1].conv2
    if arch == "densenet121":
        return model.features.denseblock4.denselayer16.conv2
    raise ValueError("unknown arch")

# ----------------------------- main --------------------------- #
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1) build + load weights
    model = build_model(args.arch, num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    layer = get_last_conv_layer(model, args.arch)
    cam_class = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "eigencam": EigenCAM}
    grad_cam = cam_class[args.method](model, layer)

    # 2) preprocess
    tfm = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    img_pil = Image.open(args.image_path).convert("RGB")  # 原图保留彩色方便叠加
    input_tensor = tfm(img_pil).unsqueeze(0).to(device)

    # 3) forward → CAM
    cam, pred_class, score = grad_cam(input_tensor)

    # 4) visualize
    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap[:, :, ::-1] / 255.0            # BGR→RGB
    overlay = heatmap * 0.5 + img_np * 0.5
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"Class {pred_class}  Score {score:.3f}")
  
    output_dir = Path(args.output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output or f"cam_{Path(args.image_path).stem}.png"
    out_path = output_dir / output_name
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f" CAM saved to {out_path}")

# ----------------------------- CLI ---------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["resnet18", "densenet121"], required=True)
    p.add_argument("--model_path", required=True, help=".pth weight file")
    p.add_argument("--image_path", required=True, help="input image to visualize")
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--method", default="gradcam", choices=["gradcam", "gradcam++", "eigencam"])
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--output", default=None, help="output PNG file name")
    args = p.parse_args()
    main(args)