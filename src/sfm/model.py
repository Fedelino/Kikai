import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .utils import get_depths_and_poses


class SfMModel(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()

        self.depth_net = smp.DeepLabV3Plus(in_channels=in_channels, encoder_name="resnext50_32x4d", encoder_weights='swsl', activation=None)
        self.pose_reduction = nn.Sequential(
            nn.Conv2d(2048, 512, (1, 1)), nn.ReLU(),  nn.BatchNorm2d(512),
        )
        self.squeeze_unsqueeze = nn.Sequential(
            nn.Conv2d(4096, 512, (1, 1)), nn.ReLU(),  nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (1, 1)), nn.ReLU(),  nn.BatchNorm2d(512),
            nn.Conv2d(512, 2048, (1, 1)), nn.ReLU(),  nn.BatchNorm2d(2048),
        )
        self.pose_decoder = nn.Sequential(
            nn.Conv2d(1024, 256, (1, 1)), nn.ReLU(),  nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3)), nn.ReLU(), nn.BatchNorm2d(256), 
            nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),  nn.BatchNorm2d(256),
            nn.Conv2d(256, 6, (3, 3), bias=False),
        )  

    def forward(self, images, intrinsics):
        """
        images: list of [B,3,H,W] tensors (length T)
        intrinsics: [B,3,3] or [3,3] tensor
        """
        features = [self.extract_features(img) for img in images]
        if intrinsics.dim() == 2:
            B = images[0].shape[0]
            intrinsics = intrinsics.unsqueeze(0).repeat(B, 1, 1)
        depths, poses, _ = self.get_depth_and_poses_from_features(images, features, intrinsics)
        return depths, poses

    def extract_features(self, x):
        return self.depth_net.encoder(x)
        
    def get_depth_and_poses_from_features(self, images, features, intrinsics):
        T  = len(images)
        B  = images[0].shape[0]

        # Collapse time into batch: [T*B, 3, H, W]
        images_bt = torch.cat(images, dim=0)

        # Each level flen: concat over time along batch -> [T*B, C, H, W]
        features_bt = [
            torch.cat([f[flen] for f in features], dim=0)
            for flen in range(len(features[0]))
        ]

        # Run depth/pose heads on flattened batch
        depth_bt, pose_bt = get_depths_and_poses(
            self.depth_net.encoder,
            self.depth_net.segmentation_head,
            self.depth_net.decoder,
            self.pose_decoder,
            images_bt,
            features_bt,
            self.pose_reduction,
            self.squeeze_unsqueeze,
        )

        # Shape back to [T, B, ...]
        # depth_bt is [T*B, 1, H, W] -> [T, B, 1, H, W]
        depth = depth_bt.view(T, B, *depth_bt.shape[1:])
        # pose_bt is typically [T*B, 6] (or [T*B, C_p, h, w] if conv); reshape accordingly:
        pose  = pose_bt.view(T, B, *pose_bt.shape[1:])

        # same depth scaling you had
        depth = (1 / (25 * torch.sigmoid(depth) + 0.1))

        # intrinsics expected per-batch; repeat over time dimension
        if intrinsics is not None:
   
            intrinsics_T = intrinsics.unsqueeze(0).expand(T, *intrinsics.shape).contiguous()
        else:
            intrinsics_T = None

        return depth, pose, intrinsics_T
