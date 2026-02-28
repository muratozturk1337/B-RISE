from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


ROOT = Path(__file__).parent.parent.parent
IMAGES = ROOT / "images"
ARTIFACTS = ROOT / "artifacts"


def resize_mask(mask, up_size, mode='bilinear'):
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,s,s)
    if mode == 'bilinear':
        mask_up = F.interpolate(mask_t, size=up_size.tolist(), mode='bilinear', align_corners=False)
    else:
        mask_up = F.interpolate(mask_t, size=up_size.tolist(), mode=mode)

    return mask_up.squeeze()

class RISE_B(nn.Module):
    def __init__(self, model, x, device=None):
        super(RISE_B, self).__init__()
        self.model = model
        _, _, H, W = x.shape
        self.input_size = (H, W)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_masks(self, N, s, p, savepath = ARTIFACTS / 'masks.npy', mode='bilinear'):
        cell_size = np.ceil(np.array(self.input_size) / s).astype(int)
        self.up_size = (s + 1) * cell_size  # +1 so we can shift grids

        grids = np.random.rand(N, s, s) < p
        self.grids = grids.astype("float32")

        shifts = []
        for _ in range(N):
            x_shift = np.random.randint(0, cell_size[0])
            y_shift = np.random.randint(0, cell_size[1])
            shifts.append((x_shift, y_shift))

        self.N = N
        self.s = s
        self.shifts = shifts
        self.mode = mode

    def render_grid(self, grid_np, shift):
        up_mask = resize_mask(grid_np, self.up_size, mode=self.mode)

        x, y = shift
        cropped = up_mask[x:x + self.input_size[0], y:y + self.input_size[1]]

        return cropped.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x, target_class):
        s = self.s
        contrib = torch.zeros(s, s, device=self.device)

        for idx in range(self.N):

            grid = self.grids[idx]
            shift = self.shifts[idx]

            # 1️⃣ Baseline
            base_mask = self.render_grid(grid, shift).to(self.device)
            with torch.no_grad():
                baseline = self.model(x * base_mask)[0, target_class]

            # 2️⃣ Find zero grid cells
            zero_positions = np.argwhere(grid == 0)

            if len(zero_positions) == 0:
                continue

            # 3️⃣ Generate flipped masks
            flipped_masks = []

            for (i, j) in zero_positions:
                flipped_grid = grid.copy()
                flipped_grid[i, j] = 1

                mask = self.render_grid(flipped_grid, shift)
                flipped_masks.append(mask)

            # Stack into batch
            flipped_masks = torch.cat(flipped_masks, dim=0).to(self.device)

            # Repeat image
            x_batch = x.repeat(flipped_masks.shape[0], 1, 1, 1)

            # 4️⃣ Batched forward
            with torch.no_grad():
                outputs = self.model(x_batch * flipped_masks)[:, target_class]

            diffs = outputs - baseline

            # 5️⃣ Accumulate grid-level contributions
            for diff, (i, j) in zip(diffs, zero_positions):
                contrib[i, j] += diff

        return contrib / self.N