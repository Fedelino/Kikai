import torch
import torch.nn.functional as F
from sfm.inverse_warp import pose_vec2mat, invert_pose

def _vec_to_3x4(vec):
    """
    vec: (B,6) -> [R|t] via euler; (B,3) -> identity R + t
         (B,1,1,6) / (B,1,1,3) etc. are squeezed automatically.
         (B,3,4)/(B,4,4) pass through.
    """
    # matrix passthroughs
    if vec.dim() == 3 and vec.shape[-2:] == (3, 4):  # (B,3,4)
        return vec
    if vec.dim() == 3 and vec.shape[-2:] == (4, 4):  # (B,4,4)
        return vec[:, :3, :]

    # squeeze stray singleton dims, e.g. (B,1,1,6) -> (B,6)
    while vec.dim() > 2 and vec.size(-2) == 1:
        vec = vec.squeeze(-2)
    while vec.dim() > 2 and vec.size(-2) == 1:
        vec = vec.squeeze(-2)
    if vec.dim() > 2 and vec.size(-1) in (3, 6):
        vec = vec.view(vec.size(0), -1)  # (B,3) or (B,6)

    if vec.shape[-1] == 6:
        return pose_vec2mat(vec)  # (B,3,4)
    if vec.shape[-1] == 3:
        Bv = vec.shape[0]
        R = torch.eye(3, device=vec.device, dtype=vec.dtype).unsqueeze(0).expand(Bv, 3, 3).contiguous()
        t = vec[:, :3].unsqueeze(-1)
        return torch.cat([R, t], dim=2)
    raise ValueError(f"pose vec must have 3 or 6 values, or be (B,3,4)/(B,4,4); got {tuple(vec.shape)}")
    
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
    images:      (B,T,3,H,W) or (T,B,3,H,W)
    depths:      (B,T,1,H,W) or (T,B,1,H,W) or (B,T,1,1,H,W)/(T,B,1,1,H,W)
    poses:       any of (B,T,6), (B,T,3), (B,T,3,4), (B,T,4,4), or with extra singleton dims like (B,T,1,1,6)
                 (absolute per-frame poses preferred; fallback path supports relative-per-neighbor with 1 line change)
    intrinsics:  (B,3,3), (B,T,3,3), (T,3,3), (3,3), (B,6|4), (B,T,6|4), (T,6|4), (6|4)
                 where vec = [fx, fy, cx, cy, (alpha), (beta)]  (alpha/beta ignored here)
    """

    def loss_fn(images, depths, poses, intrinsics):
        # ---------- normalize images/depths to (B,T,...) ----------
        if images.dim() != 5 or depths.dim() not in (5, 6):
            raise ValueError(
                f"Expected images 5D and depths 5D/6D, got images {images.dim()}D {tuple(images.shape)}, "
                f"depths {depths.dim()}D {tuple(depths.shape)}"
            )

        # images (T,B,3,H,W) -> (B,T,3,H,W)
        if images.size(2) == 3 and images.size(0) == depths.size(1) and images.size(1) == depths.size(0):
            images = images.permute(1, 0, 2, 3, 4).contiguous()

        # depths: (B,T,1,1,H,W)/(T,B,1,1,H,W) -> (B,T,1,H,W)
        if depths.dim() == 6:
            if depths.size(0) == images.size(1) and depths.size(1) == images.size(0):
                depths = depths.permute(1, 0, 2, 3, 4, 5).contiguous()
            if depths.size(3) == 1:
                depths = depths.squeeze(3)
            elif depths.size(2) == 1:
                depths = depths.squeeze(2)

        # depths (T,B,1,H,W) -> (B,T,1,H,W)
        if depths.dim() == 5 and depths.size(0) == images.size(1) and depths.size(1) == images.size(0):
            depths = depths.permute(1, 0, 2, 3, 4).contiguous()

        assert images.dim() == 5 and depths.dim() == 5
        assert images.size(0) == depths.size(0) and images.size(1) == depths.size(1)

        B, T, C, H, W = images.shape
        device = images.device
        dtype  = images.dtype

        # ---------- normalize poses to something usable ----------
        if poses is not None:
            # time-major -> batch-major
            if poses.dim() >= 3 and poses.size(0) == images.size(1) and poses.size(1) == images.size(0):
                poses = poses.permute(1, 0, *range(2, poses.dim())).contiguous()
            # common in your model: (B,T,1,1,6 or 3) -> (B,T,6 or 3)
            if poses.dim() == 5 and poses.size(2) == 1 and poses.size(3) == 1 and poses.size(4) in (3, 6):
                poses = poses.squeeze(3).squeeze(2).contiguous()  # (B,T,3) or (B,T,6)

        target_idx   = T // 2
        target_img   = images[:, target_idx].contiguous().float()   # (B,3,H,W)
        target_depth = depths[:, target_idx].contiguous().float()   # (B,1,H,W)

        # ---------- intrinsics -> K (B,3,3) for target ----------
        def _build_K_from_vec(vec):  # vec: (B,4|6)
            fx, fy, cx, cy = vec[:, 0], vec[:, 1], vec[:, 2], vec[:, 3]
            K = torch.zeros(vec.size(0), 3, 3, device=vec.device, dtype=vec.dtype)
            K[:, 0, 0] = fx; K[:, 1, 1] = fy
            K[:, 0, 2] = cx; K[:, 1, 2] = cy
            K[:, 2, 2] = 1.0
            return K

        def _select_K(I, t_idx, B):
            if I is None:
                raise ValueError("intrinsics is None")
            # matrices
            if I.dim() >= 3 and I.shape[-2:] == (3, 3):
                if I.dim() == 4:   # (B,T,3,3)
                    return I[:, t_idx].to(device=device, dtype=target_img.dtype)
                if I.dim() == 3:   # (T,3,3)
                    return I[t_idx].unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=target_img.dtype)
                if I.dim() == 2:   # (3,3)
                    return I.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=target_img.dtype)
            # vectors
            if I.dim() == 3 and I.size(-1) in (4, 6):     # (B,T,4|6)
                return _build_K_from_vec(I[:, t_idx]).to(device=device, dtype=target_img.dtype)
            if I.dim() == 2 and I.size(-1) in (4, 6):     # (B,4|6) or (T,4|6)
                if I.size(0) == B:
                    return _build_K_from_vec(I).to(device=device, dtype=target_img.dtype)
                if I.size(0) == T:
                    vec = I[t_idx].unsqueeze(0).expand(B, -1)
                    return _build_K_from_vec(vec).to(device=device, dtype=target_img.dtype)
            if I.dim() == 1 and I.numel() in (4, 6):      # (4|6)
                vec = I.unsqueeze(0).expand(B, -1)
                return _build_K_from_vec(vec).to(device=device, dtype=target_img.dtype)
            raise ValueError(f"Unsupported intrinsics shape {tuple(I.shape)}")

        K    = _select_K(intrinsics, target_idx, B)   # (B,3,3)
        Kinv = torch.inverse(K)

        # ---------- pixel grid & backprojection (target cam) ----------
        u = torch.linspace(0, W - 1, W, device=device, dtype=target_img.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        v = torch.linspace(0, H - 1, H, device=device, dtype=target_img.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        ones = torch.ones_like(u)
        pix = torch.cat([u, v, ones], dim=1).view(B, 3, -1)  # (B,3,H*W)

        D  = target_depth.view(B, 1, -1)                     # (B,1,H*W)
        Xc = (Kinv @ pix) * D                                # (B,3,H*W)

        # ---------- helper: convert various pose formats to [R|t] (B,3,4) ----------
        def _vec_to_3x4(vec):
            """
            vec: (B,6) -> [R|t] via euler; (B,3) -> identity R + t
                 (B,3,4)/(B,4,4) pass-through; also squeezes stray singleton dims.
            """
            # matrix pass-throughs
            if vec.dim() == 3 and vec.shape[-2:] == (3, 4):
                return vec
            if vec.dim() == 3 and vec.shape[-2:] == (4, 4):
                return vec[:, :3, :]

            # squeeze cases like (B,1,1,6) -> (B,6)
            while vec.dim() > 2 and vec.size(-2) == 1:
                vec = vec.squeeze(-2)
            while vec.dim() > 2 and vec.size(-2) == 1:
                vec = vec.squeeze(-2)
            if vec.dim() > 2 and vec.size(-1) in (3, 6):
                vec = vec.view(vec.size(0), -1)

            if vec.shape[-1] == 6:
                return pose_vec2mat(vec)  # (B,3,4)
            if vec.shape[-1] == 3:
                Bv = vec.shape[0]
                R = torch.eye(3, device=vec.device, dtype=vec.dtype).unsqueeze(0).expand(Bv, 3, 3).contiguous()
                t = vec[:, :3].unsqueeze(-1)
                return torch.cat([R, t], dim=2)
            raise ValueError(f"pose vec must have 3 or 6 values, or be (B,3,4)/(B,4,4); got {tuple(vec.shape)}")

        # ---------- absolute poses -> (B,T,4,4) if available ----------
        T_abs = None
        if poses is not None:
            # already matrices?
            if poses.dim() == 5 and poses.shape[-2:] in [(3, 4), (4, 4)]:
                mats = []
                for t in range(T):
                    m = poses[:, t]
                    if m.shape[-2:] == (4, 4):
                        m = m[:, :3, :]
                    mats.append(m)  # (B,3,4)
            else:
                # vectors (B,T,6 or 3)
                mats = [_vec_to_3x4(poses[:, t]) for t in range(T)]  # list of (B,3,4)

            T_abs = torch.stack([
                torch.cat([m, torch.tensor([0, 0, 0, 1], device=m.device, dtype=m.dtype)
                          .view(1, 1, 4).expand(m.size(0), 1, 4)], dim=1)
                for m in mats
            ], dim=1)  # (B,T,4,4)

        # ---------- photometric helper ----------
        def _photo(a, b):
            if with_ssim:
                l1 = (a - b).abs().mean(1, keepdim=True)
                s  = ssim(a, b)  # if similarity in [0,1], turn into loss
                s_loss = 1.0 - s if s.min() >= 0 and s.max() <= 1 else s
                return 0.15 * l1 + 0.85 * s_loss
            else:
                return (a - b).abs().mean(1, keepdim=True)

        warped_losses, identity_losses, reproj_dbg = [], [], []

        # ---------- neighbors: warp neighbor -> target ----------
        for offset in range(-neighbor_range, neighbor_range + 1):
            if offset == 0:
                continue
            n_idx = target_idx + offset
            if not (0 <= n_idx < T):
                continue

            neighbor_img = images[:, n_idx].float()  # (B,3,H,W)

            # Relative pose T_tn: target -> neighbor
            if T_abs is not None:
                T_t     = T_abs[:, target_idx]
                T_n     = T_abs[:, n_idx]
                T_t_inv = invert_pose(T_t)
                T_tn    = T_n @ T_t_inv
            else:
                # Fallback if your model emits relative poses per neighbor:
                # map 'offset' to j; adapt this one line to your layout if needed.
                j = offset + neighbor_range - (1 if offset > 0 else 0)
                T_3x4 = _vec_to_3x4(poses[:, j])
                T_tn  = torch.eye(4, device=device).expand(B, 4, 4).clone()
                T_tn[:, :3, :4] = T_3x4

            # Transform target 3D points into neighbor camera
            R  = T_tn[:, :3, :3]
            t  = T_tn[:, :3,  3:4]
            Xn = R @ Xc + t                                   # (B,3,H*W)

            # Project with K and sample neighbor
            x   = K @ Xn                                      # (B,3,H*W)
            z   = x[:, 2:3, :].clamp(min=1e-6)
            x   = x / z
            u2  = x[:, 0, :].view(B, 1, H, W)
            v2  = x[:, 1, :].view(B, 1, H, W)
            grid_x = 2.0 * (u2 / (W - 1.0)) - 1.0
            grid_y = 2.0 * (v2 / (H - 1.0)) - 1.0
            grid   = torch.stack([grid_x, grid_y], dim=-1).squeeze(1)  # (B,H,W,2)

            warped = F.grid_sample(
                neighbor_img, grid, mode="bilinear",
                padding_mode=padding_mode, align_corners=False
            )  # (B,3,H,W)

            warped_losses.append(_photo(target_img, warped))
            if with_auto_mask:
                identity_losses.append(_photo(target_img, neighbor_img))
            if return_reprojections and offset == 1:
                reproj_dbg.append(warped.detach())

        if len(warped_losses) == 0:
            photometric_loss = torch.tensor(0.0, device=device)
        else:
            vol = torch.stack(warped_losses, dim=1)  # (B,Nbr,1,H,W)
            if with_auto_mask and len(identity_losses) == len(warped_losses):
                id_vol = torch.stack(identity_losses, dim=1)
                vol = torch.cat([vol, id_vol], dim=1)
            photometric_loss = vol.min(dim=1).values.mean()

        # edge-aware smoothness (on target)
        smoothness_loss = edge_aware_smoothness_loss(target_depth, target_img)
        geometric_consistency_loss = torch.tensor(0.0, device=device)

        if not return_reprojections:
            return (
                photometric_loss_weight * photometric_loss,
                geometric_consistency_loss_weight * geometric_consistency_loss,
                smoothness_loss_weight * smoothness_loss
            )
        else:
            reproj_sample = reproj_dbg[0] if len(reproj_dbg) else None
            return (
                photometric_loss_weight * photometric_loss,
                geometric_consistency_loss_weight * geometric_consistency_loss,
                smoothness_loss_weight * smoothness_loss,
                reproj_sample
            )

    return loss_fn

def l2_pose_regularization(poses):
    l2loss = []
    for pose in poses:
        for p in pose:
            if len(p)>0:
                l2loss.append((p[0]**2).mean())
    return sum(l2loss) / len(l2loss)
