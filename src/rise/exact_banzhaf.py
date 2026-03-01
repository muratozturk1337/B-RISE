import torch


def build_supergrid_mask(mask_vec, s=2, H=28, W=28):
    block_h = H // s
    block_w = W // s

    mask = torch.zeros(1, 1, H, W)

    for i in range(s * s):
        if mask_vec[i] == 1:
            r = i // s
            c = i % s

            mask[:, :, 
                 r*block_h:(r+1)*block_h,
                 c*block_w:(c+1)*block_w] = 1

    return mask


def exact_banzhaf_supergrid(model, x, s):
    if s > 4:
        raise ValueError("s too large, number of subsets grows exponentially")
    
    model.eval()
    device = x.device

    H, W = x.shape[-2:]
    d = s * s
    num_subsets = 2**d
    num_classes = model(x).shape[1]

    values = torch.zeros(num_subsets, num_classes, device=device)

    # Precompute P(S)
    with torch.no_grad():
        for mask_int in range(num_subsets):

            mask_vec = torch.tensor(
                [(mask_int >> j) & 1 for j in range(d)],
                device=device,
                dtype=torch.float32
            )

            mask_S = build_supergrid_mask(mask_vec, s, H, W).to(device)
            values[mask_int] = model(x * mask_S).squeeze(0)

    # Compute exact Banzhaf
    beta = torch.zeros(num_classes, d, device=device)

    for i in range(d):
        total = torch.zeros(num_classes, device=device)

        for mask_int in range(num_subsets):

            if (mask_int >> i) & 1:
                continue

            S = mask_int
            Si = mask_int | (1 << i)

            total += values[Si] - values[S]

        beta[:, i] = total / (2**(d-1))

    return beta.view(num_classes, s, s)