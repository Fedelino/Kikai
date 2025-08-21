import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from torch import nn
import torch


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.45 + tensor.numpy()*0.225
    return array


def change_bn_momentum(model, new_value):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = new_value


def get_depths_and_poses(
    encoder, segmentation_head, decoder, pose_decoder,
    images, features_, reduction, squeeze_unsqueeze
):
    import torch

    # ---------- Normalize IMAGES to 4D (B*T, C, H, W) ----------
    if hasattr(images, "dim"):
        if images.dim() == 5:
            b, l, c, h, w = images.shape                  # (B, T, C, H, W)
            images_bt = images.flatten(0, 1)              # (B*T, C, H, W)
        elif images.dim() == 4:
            bt, c, h, w = images.shape                     # (B*T, C, H, W)
            # try to infer (B, T) from features_ if they are 5D
            if hasattr(features_[-1], "dim") and features_[-1].dim() == 5:
                b, l, _, _, _ = features_[-1].shape
                assert b * l == bt, "Inconsistent (B,T) vs images batch"
            else:
                # fallback: treat as T=1 (no temporal dimension)
                b, l = bt, 1
            images_bt = images
        else:
            raise ValueError("images must be a 4D or 5D tensor")
    else:
        raise ValueError("images must be a torch.Tensor")

    # ---------- Normalize FEATURES to 4D list ----------
    if not isinstance(features_, (list, tuple)) or len(features_) == 0:
        raise ValueError("features_ must be a non-empty list/tuple of tensors")

    if hasattr(features_[-1], "dim") and features_[-1].dim() == 5:
        # Each level (B, T, C, H, W) -> (B*T, C, H, W)
        feats_bt = [f.flatten(0, 1) for f in features_]
        _, c_feat, h_feat, w_feat = feats_bt[-1].shape
    else:
        feats_bt = features_
        _, c_feat, h_feat, w_feat = feats_bt[-1].shape

    # ---------- Rebuild "ref_features" & "features" like your original code ----------
    # We need the 5D view (B, T, C, H, W) briefly to replicate the logic.
    # Convert feats_bt back to (B, T, C, H, W) for this manipulation.
    feats_5d = [f.view(b, l, f.shape[1], f.shape[2], f.shape[3]) for f in feats_bt]

    # ref_features: first B along the (B,T) batch is just the whole batch; keep semantics as in your code
    ref_features = [x[:, :] for x in feats_5d]  # same as x[:b] in 5D since dim0==B

    features = []
    lf = len(feats_5d)

    for i in range(lf):
        if i == lf - 1:
            # last level gets concatenation with "reference-expanded" then squeeze/unsqueeze
            last = feats_5d[-1]  # (B, T, C, H, W)
            ref_last = ref_features[-1]  # (B, T, C, H, W)

            # expand ref along time and combine like your original:
            ref_expanded = ref_last.reshape(b, 1, c_feat, h_feat, w_feat) \
                                   .expand(b, l, c_feat, h_feat, w_feat) \
                                   .reshape(b * l, c_feat, h_feat, w_feat)  # (B*T, C, H, W)

            last_bt = last.reshape(b * l, c_feat, h_feat, w_feat)           # (B*T, C, H, W)

            mixed = torch.cat([last_bt, ref_expanded], dim=1)               # (B*T, 2C, H, W)
            mixed = squeeze_unsqueeze(mixed)                                 # keep your op
            features.append(mixed)                                           # (B*T, 2C, H, W) after squeeze/unsqueeze
        else:
            # other levels just pass through
            features.append(feats_bt[i])  # (B*T, C, H, W)

    # ---------- Depth head ----------
    # decoder(features) -> (B*T, C_d, H, W), seg-head -> (B*T, 1, H, W)
    depths_bt = segmentation_head(decoder(features))                         # (B*T, 1, H, W)
    depths = depths_bt.view(b, l, 1, h, w)                                   # (B, T, 1, H, W)

    # ---------- Pose head ----------
    last_feat = features[-1]                                                 # (B*T, C*, Hf, Wf)
    # reduce channels/spatial as your code
    last_feat_red = reduction(last_feat)                                     # (B*T, C', Hf, Wf)
    c_reduced = last_feat_red.shape[1]
    last_feat_5d = last_feat_red.view(b, l, c_reduced, h_feat, w_feat)       # (B, T, C', Hf, Wf)

    # build pairwise (i,j) feature grid: (B, T, T, 2C', Hf, Wf) -> (B*T*T, 2C', Hf, Wf)
    last_i = last_feat_5d.unsqueeze(2).expand(b, l, l, c_reduced, h_feat, w_feat)   # (B, T, T, C', Hf, Wf)
    features_sq = torch.cat([last_i, last_i.transpose(1, 2)], dim=3)                # concat on channel: 2C'
    features_sq = features_sq.reshape(b * l * l, 2 * c_reduced, h_feat, w_feat)     # (B*T*T, 2C', Hf, Wf)

    poses_logits = pose_decoder(features_sq)                                        # (B*T*T, 6, Hf, Wf)
    poses = poses_logits.mean(dim=(2, 3)).reshape(b, l, l, 6)                       # (B, T, T, 6)

    return depths, poses * 0.01
