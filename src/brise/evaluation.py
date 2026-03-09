import torch
import torch.nn.functional as F
import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt


def auc(arr):
    return trapezoid(arr) / (len(arr) - 1)

def zero_substrate(x):
    return torch.zeros_like(x)

def blur_substrate(x):
    return F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)

def evaluate_saliency(model, image, saliency, mode="del", step=100, target_class=None, progress_image_count=0, substrate_fn=zero_substrate):
    assert mode in ["del", "ins"], "mode must be 'del' or 'ins'"

    device = next(model.parameters()).device
    image = image.to(device)

    # Get target class if not provided
    if target_class is None:
        with torch.no_grad():
            pred = model(image)
            target_class = pred.argmax(dim=1).item()

    H, W = saliency.shape
    HW = H * W

    # Flatten saliency
    flat_sal = saliency.flatten()

    # Sort descending (RAW values, no abs)
    order = np.argsort(-flat_sal)

    # Number of steps
    n_steps = (HW + step - 1) // step

    # Setup insertion / deletion
    if mode == "del":
        start = image.clone()
        finish = substrate_fn(image)
    else:
        start = substrate_fn(image)
        finish = image.clone()

    scores = [] 
    C = start.shape[1]

    for i in range(n_steps + 1):

        # Evaluate
        with torch.no_grad():
            pred = model(start)
            prob = torch.softmax(pred, dim=1)[0, target_class].item()
        scores.append(prob)

        if progress_image_count > 0:
            interval = max(1, n_steps // progress_image_count)

            if i in [s for s in range(0, n_steps + 1, interval)]:  # visualize start, middle, end
                img_vis = start[0].detach().cpu()

                # Undo normalization for display (ImageNet)
                if C == 3:
                    # ImageNet unnormalize
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                    img_vis = img_vis * std + mean
                    img_vis = img_vis.clamp(0,1)
                    plt.imshow(img_vis.permute(1,2,0))

                elif C == 1:
                    # MNIST
                    plt.imshow(img_vis[0], cmap="gray")

                plt.title(f"Step {i}")
                plt.axis("off")
                plt.show()
        
        if i == n_steps:
            break
        
        # Pixels to modify this step
        coords = order[i*step:(i+1)*step]

        # Convert tensor to numpy view
        
        start_np = start.cpu().numpy().reshape(1, C, HW)
        finish_np = finish.cpu().numpy().reshape(1, C, HW)

        # Replace pixels
        start_np[0, :, coords] = finish_np[0, :, coords]

        # Move back to tensor
        start = torch.from_numpy(start_np.reshape(1, C, H, W)).to(device)

    scores = np.array(scores)
    return scores, auc(scores)