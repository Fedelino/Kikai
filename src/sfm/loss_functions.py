import torch
import torch.nn.functional as F

def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)
    sigma1 = F.avg_pool2d(img1 ** 2, 3, 1, 1) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2 ** 2, 3, 1, 1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2

    ssim_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    ssim_map = ssim_n / ssim_d
    return torch.clamp((1 - ssim_map) / 2, 0, 1)  # SSIM distance

def _collapse_time(x):
    """
    If x is 5-D, pick the middle time frame and return 4-D (B,C,H,W).
    Handles both (B,T,C,H,W) and (T,B,C,H,W).
    """
    if x.dim() == 5:
        if x.size(2) in (1, 3):            # (B,T,C,H,W)
            t = x.size(1) // 2
            x = x[:, t]                    # -> (B,C,H,W)
        elif x.size(0) > 1 and x.size(2) in (1, 3):  # (T,B,C,H,W)
            t = x.size(0) // 2
            x = x[t]                        # -> (B,C,H,W)
        else:
            # fallback: squeeze any singleton time-like dim
            x = x.squeeze(1).squeeze(0)
    return x

def edge_aware_smoothness_loss(depth, image):
    # --- Normalize to 4-D (B,C,H,W), collapse time if present ---
    depth = _collapse_time(depth)
    image = _collapse_time(image)

    # Ensure channel dims
    if depth.dim() == 3: depth = depth.unsqueeze(1)   # (B,H,W)->(B,1,H,W)
    if image.dim() == 3: image = image.unsqueeze(1)   # (B,H,W)->(B,1,H,W)

    # Resize depth to image size if needed
    if depth.shape[-2:] != image.shape[-2:]:
        depth = F.interpolate(depth, size=image.shape[-2:], mode="bilinear", align_corners=False)

    # Grayscale for edge weights (channel dim is now dim=1)
    img_gray = image.mean(dim=1, keepdim=True) if image.size(1) > 1 else image

    # Finite differences with matched slicing
    depth_grad_x = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])   # (B,1,H,W-1)
    depth_grad_y = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])   # (B,1,H-1,W)
    img_grad_x   = torch.abs(img_gray[:, :, :, 1:] - img_gray[:, :, :, :-1])
    img_grad_y   = torch.abs(img_gray[:, :, 1:, :] - img_gray[:, :, :-1, :])

    weight_x = torch.exp(-img_grad_x)
    weight_y = torch.exp(-img_grad_y)

    return (depth_grad_x * weight_x).mean() + (depth_grad_y * weight_y).mean()
def get_all_loss_fn(
    neighbor_range,
    subsampled_sequence_length,
    photometric_loss_weight,
    geometric_consistency_loss_weight,
    smoothness_loss_weight,
    with_ssim,
    with_mask,
    with_auto_mask,
    padding_mode,
    return_reprojections=False
):
    """
    Returns a callable loss function configured with the provided arguments.
    """
    def loss_fn(images, depths, poses, intrinsics):
        if images.dim() != 5 or depths.dim() not in (5, 6):
            raise ValueError(
                f"Expected images 5D and depths 5D/6D, got images {images.dim()}D {tuple(images.shape)}, "
                f"depths {depths.dim()}D {tuple(depths.shape)}"
            )

        # images is (T,B,3,H,W) -> (B,T,3,H,W)
        if images.size(2) == 3 and images.size(0) == depths.size(1) and images.size(1) == depths.size(0):
            images = images.permute(1, 0, 2, 3, 4).contiguous()

        # depths could be (B,T,1,1,H,W) or (T,B,1,1,H,W) -> drop extra singleton & permute if needed
        if depths.dim() == 6:
            # If (T,B,1,1,H,W), first permute to (B,T,1,1,H,W)
            if depths.size(0) == images.size(1) and depths.size(1) == images.size(0):
                depths = depths.permute(1, 0, 2, 3, 4, 5).contiguous()
            # Now squeeze the extra middle 1: (B,T,1,1,H,W) -> (B,T,1,H,W)
            if depths.size(3) == 1:
                depths = depths.squeeze(3)
            elif depths.size(2) == 1:
                depths = depths.squeeze(2)

        # depths is (T,B,1,H,W) -> (B,T,1,H,W)
        if depths.dim() == 5 and depths.size(0) == images.size(1) and depths.size(1) == images.size(0):
            depths = depths.permute(1, 0, 2, 3, 4).contiguous()

        # ---- Sanity checks ----
        assert images.dim() == 5 and depths.dim() == 5, \
            f"After normalize: images {tuple(images.shape)}, depths {tuple(depths.shape)}"
        assert images.size(0) == depths.size(0) and images.size(1) == depths.size(1), \
            f"B/T mismatch: images {tuple(images.shape)} vs depths {tuple(depths.shape)}"
        B, T, _, H, W = images.shape
        target_idx = T // 2
        target_img   = images[:, target_idx].contiguous().float()   # (B,3,H,W)
        target_depth = depths[:, target_idx].contiguous().float() 
        photometric_loss = 0.0
        smoothness_loss = 0.0
        geometric_consistency_loss = 0.0

        target_idx = T // 2  # middle frame
        target_img = images[:, target_idx]
        target_depth = depths[:, target_idx]

        # Iterate neighbors
        for offset in range(-neighbor_range, neighbor_range + 1):
            if offset == 0 or not (0 <= target_idx + offset < T):
                continue
            neighbor_img = images[:, target_idx + offset]

            # In a real pipeline: warp neighbor -> target view here
            reproj = neighbor_img  # placeholder — should be warped image

            # Photometric loss
            if with_ssim:
                loss_photo = 0.85 * ssim(target_img, reproj).mean() + 0.15 * torch.abs(target_img - reproj).mean()
            else:
                loss_photo = torch.abs(target_img - reproj).mean()
            photometric_loss += loss_photo

        # Smoothness loss
        smoothness_loss = edge_aware_smoothness_loss(target_depth, target_img)

        # Geometric consistency loss placeholder
        geometric_consistency_loss = torch.tensor(0.0, device=images.device)

        return (
            photometric_loss_weight * photometric_loss,
            geometric_consistency_loss_weight * geometric_consistency_loss,
            smoothness_loss_weight * smoothness_loss
        )

    return loss_fn

def l2_pose_regularization(poses):
    l2loss = []
    for pose in poses:
        for p in pose:
            if len(p)>0:
                l2loss.append((p[0]**2).mean())
    return sum(l2loss) / len(l2loss)
