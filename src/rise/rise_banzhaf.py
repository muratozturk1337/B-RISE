import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


class RISE_B(nn.Module):
    def __init__(self, model, x, gpu_batch=100, device=None):
        super(RISE_B, self).__init__()
        self.model = model
        _, _, H, W = x.shape
        self.input_size = (H, W)
        self.gpu_batch = gpu_batch
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_masks(self, N, s, p, mode='bilinear'):
        cell_size = np.ceil(np.array(self.input_size) / s).astype(int)
        self.up_size = (s + 1) * cell_size  # +1 so we can shift grids

        grids = np.random.rand(N, s, s) < p
        self.grids = grids.astype("float32")

        self.shifts = []
        for _ in range(N):
            x_shift = np.random.randint(0, cell_size[0])
            y_shift = np.random.randint(0, cell_size[1])
            self.shifts.append((x_shift, y_shift))

        self.N = N
        self.s = s
        self.p = p
        self.mode = mode

    def resize_mask(self, mask, up_size, mode='bilinear'):
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,s,s)
        if mode == 'bilinear':
            mask_up = F.interpolate(mask_t, size=up_size.tolist(), mode='bilinear', align_corners=False)
        else:
            mask_up = F.interpolate(mask_t, size=up_size.tolist(), mode=mode)

        return mask_up.squeeze()
    
    def render_grid(self, grid_np, shift):
        up_mask = self.resize_mask(grid_np, self.up_size, mode=self.mode)

        x, y = shift
        cropped = up_mask[x:x + self.input_size[0], y:y + self.input_size[1]]

        return cropped.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        with torch.no_grad():
                num_classes = self.model(x).shape[1]

        H, W = self.input_size
        saliency = torch.zeros(num_classes, H, W, device=self.device)
        forwardpass_counter = 0

        for idx in tqdm(range(self.N), desc="Banzhaf sampling"):
            
            grid = self.grids[idx]
            shift = self.shifts[idx]

            base_mask = self.render_grid(grid, shift).to(self.device)       # (1,1,H,W)
            with torch.no_grad():
                baseline = self.model(x * base_mask)                    # (1,num_classes)
                forwardpass_counter += 1
                
            zero_positions = np.argwhere(grid == 0)
            if len(zero_positions) == 0:
                continue
            
            # calculate flipped masks for all zero positions
            flipped_masks = []

            for (i, j) in zero_positions:
                flipped_grid = grid.copy()
                flipped_grid[i, j] = 1

                mask = self.render_grid(flipped_grid, shift)
                flipped_masks.append(mask)

            # Stack into batch
            flipped_masks = torch.cat(flipped_masks, dim=0).to(self.device)     # (num_flips,1,H,W)

            # Batch forward
            with torch.no_grad():
                outputs = self.model(x * flipped_masks)
                forwardpass_counter += flipped_masks.shape[0]

            diffs = outputs - baseline              # (num_flips, num_classes)

            for idx, diff_vec in enumerate(diffs):
                delta_mask = flipped_masks[idx] - base_mask             # (1,1,H,W)
                delta_mask = delta_mask.squeeze(0)                      # (1,H,W)

                saliency += diff_vec.view(-1, 1, 1) * delta_mask        # (num_classes,H,W)

            
        print("Total forward passes:", forwardpass_counter)
        return saliency / (self.N * (1 - self.p))