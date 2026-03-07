from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).parent.parent.parent
IMAGES = ROOT / "images"

### Image utils

def load_image(path, device, preprocess):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    return img, x

def draw_image(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    plt.imshow(img)
    plt.axis("off")
    plt.show()

def draw_mask(mask):
    if torch.is_tensor(mask):
        mask = mask.cpu().squeeze().numpy()
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.show()

def draw_masked_image(img, mask):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    if torch.is_tensor(mask):
        mask = mask.cpu().permute(1, 2, 0).numpy()  # (H, W, 1)

    masked = img * mask

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # left: mask
    axes[0].imshow(mask.squeeze(), cmap="gray")
    axes[0].set_title("Mask")
    axes[0].axis("off")

    # right: masked image
    axes[1].imshow(masked)
    axes[1].set_title("Masked image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


### RISE Helper functions

def get_topk_predictions(model, x, weights, k=5):
        model.eval()

        with torch.no_grad():
            logits = model(x.float())
            values, indices = logits.topk(k, dim=1)

        values = values.squeeze(0).cpu()
        indices = indices.squeeze(0).cpu()

        topk = []
        for idx, val in zip(indices, values):
            topk.append({
                "class_id": int(idx.item()),
                "class_name": weights.meta["categories"][idx],
                "logit": float(val.item())
            })

        return topk


def draw_saliency_overlay(img, saliency_maps, topk, k=0, norm_mode="symmetric", alpha = 0.7):
    assert norm_mode in ["symmetric", "minmax"], "norm_mode must be 'symmetric' or 'minmax'"
    
    # Convert image
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    # Select class
    class_id = topk[k]["class_id"]
    class_name = topk[k]["class_name"]
    score = topk[k]["logit"]

    sal = saliency_maps[class_id]
    if torch.is_tensor(sal):
        sal = sal.cpu().numpy()

    if norm_mode == "symmetric":
        max_abs = np.abs(sal).max()
        vmin, vmax = -max_abs, max_abs
        cmap = "seismic"
    elif norm_mode == "minmax":
        vmin, vmax = sal.min(), sal.max()
        cmap = "jet"


    # Plot
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.imshow(sal, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.045, pad=0.05)
    plt.title(f"Class: {class_name} ({alpha} alpha)", fontsize=14)
    plt.axis("off")
    plt.show()

    return sal

def draw_saliency_overlay_mnist(img, saliency_maps, class_id, norm_mode="symmetric", alpha = 0.7):
    assert norm_mode in ["symmetric", "minmax"], "norm_mode must be 'symmetric' or 'minmax'"

    # Image
    if torch.is_tensor(img):
        img = img.cpu().squeeze().numpy()
    
    # Saliency
    sal = saliency_maps[class_id]
    if torch.is_tensor(sal):
        sal = sal.cpu().numpy()
    
    if norm_mode == "symmetric":
        max_abs = np.abs(sal).max()
        vmin, vmax = -max_abs, max_abs
        cmap = "seismic"
    elif norm_mode == "minmax":
        vmin, vmax = sal.min(), sal.max()
        cmap = "jet"

    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap="gray")
    plt.imshow(sal, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.045, pad=0.05)
    plt.title(f"Class {class_id} ({alpha} alpha)")
    plt.axis("off")
    plt.show()

    return sal