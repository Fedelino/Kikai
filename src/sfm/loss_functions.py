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

def edge_aware_smoothness_loss(depth, image):
    depth_grad_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    depth_grad_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
    image_grad_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
    image_grad_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)
    weight_x = torch.exp(-image_grad_x)
    weight_y = torch.exp(-image_grad_y)
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
        # Here, images: [B,T,3,H,W], depths: [B,T,1,H,W]
        B, T, _, H, W = images.shape
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
