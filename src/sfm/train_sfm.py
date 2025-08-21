import torch
from sfm.datasets import SequenceDataset
import sfm.custom_transforms as custom_transforms
from tqdm import tqdm
from sfm.model import SfMModel
from sfm.loss_functions import get_all_loss_fn, l2_pose_regularization
import argparse
import wandb
import numpy as np
import torch.nn as nn
from sfm.utils import tensor2array, change_bn_momentum
import torchvision

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
parser = argparse.ArgumentParser(description='Structure from Motion Learner training on video sequences',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', type=str, default='data/sequences', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers to load data')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--long_sequence_length', type=int, default=2, help='length of long sequences')
parser.add_argument('--subsampled_sequence_length', type=int, default=2, help='length of subsampled sequences')
parser.add_argument('--with_replacement', action='store_true', default=False, help='use replacement when subsampling')
parser.add_argument('--with_ssim', action='store_true', help='use SSIM loss')
parser.add_argument('--with_mask', action='store_true', help='use mask loss')
parser.add_argument('--with_auto_mask', action='store_true', help='use auto mask loss')
parser.add_argument('--padding_mode', type=str, default='zeros', help='padding mode for inverse warp')
parser.add_argument('--neighbor_range', type=int, default=1, help='neighbor range for pairwise loss')
parser.add_argument('--photometric_loss_weight', type=float, default=1.0, help='weight for photometric loss')
parser.add_argument('--geometric_consistency_loss_weight', type=float, default=0.5, help='weight for geometric consistency loss')
parser.add_argument('--smoothness_loss_weight', type=float, default=0.1, help='weight for smoothness loss')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--name', type=str, default='default', help='name of the experiment')
parser.add_argument('--checkpoint', type=str, help='path to checkpoint to continue training from ')
parser.add_argument('--l2_pose_reg_weight', type=float, default=0.0, help='weight for l2 pose regularization loss')
parser.add_argument('--accumulate_steps', type=int, default=1, help='number of steps to accumulate gradients over')
parser.add_argument('--intrinsics_file', default='example_inputs/intrinsics.json', help='path to intrinsics file')

def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_wandb = wandb.init(
        project="kikai",
        name=args.name,
        config=vars(args)
    )

    compute_loss = get_all_loss_fn(
        args.neighbor_range,
        args.subsampled_sequence_length,
        args.photometric_loss_weight,
        args.geometric_consistency_loss_weight,
        args.smoothness_loss_weight,
        args.with_ssim, 
        args.with_mask, 
        args.with_auto_mask, 
        args.padding_mode,
        return_reprojections=False
    )
    train_transform = custom_transforms.Compose([
        custom_transforms.ResizeImagesOnly((192, 320)), 
        custom_transforms.ColorJitter(0.10, 0.10, 0.10, 0.05),
        custom_transforms.RandomHorizontalFlip(),
    ])
    individual_transform = torchvision.transforms.ColorJitter(0.07, 0.07, 0.07, 0.03)
    train_dataset = SequenceDataset(args.data, train=True, transform=train_transform, individual_transform=individual_transform, 
                                                seed=args.seed, long_sequence_length=args.long_sequence_length, subsampled_sequence_length=args.subsampled_sequence_length, with_replacement=args.with_replacement)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_transform = custom_transforms.Compose([
        custom_transforms.ResizeImagesOnly((192, 320)),
    ])
    val_dataset = SequenceDataset(
        args.data, train=False, transform=val_transform,
        seed=args.seed,
        long_sequence_length=args.subsampled_sequence_length,
        subsampled_sequence_length=args.subsampled_sequence_length,
        with_replacement=False
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # by_me intrinsics = intrinsics.to(device)

    model = SfMModel(in_channels=3).to(device)
    if args.checkpoint is not None:
        # Only set weights_only=False if you TRUST this file
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        # Handle common checkpoint formats
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]

        # In case keys are prefixed (e.g., 'module.' from DataParallel)
        if len(ckpt):
            first_key = next(iter(ckpt))
            if first_key.startswith("module."):
                ckpt = {k[len("module."):]: v for k, v in ckpt.items()}

        model.load_state_dict(ckpt, strict=False)
    
    change_bn_momentum(model,  0.01)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = 100000

    val_every = 1 
    
    num_train_batches = len(train_loader)
    num_val_batches   = len(val_loader)

    num_val_passes = (args.epochs // val_every)

    total_steps = args.epochs * num_train_batches + num_val_passes * num_val_batches

    master_pbar = tqdm(total=total_steps, desc="Training (all epochs)", position=0)

    for epoch in range(args.epochs):

        model.train()

        for batch_idx, (images, images_individually_jittered, intrinsics) in tqdm(enumerate(train_loader)):
            images_individually_jittered = [img.to(device) for img in images_individually_jittered]
            intrinsics = intrinsics.to(device)
            updated_intrinsics = intrinsics

            depths, poses = model(images_individually_jittered, intrinsics)
            del images_individually_jittered
            images = [img.to(device) for img in images]

            if isinstance(images, list):
                images = torch.stack(images, dim=1)

            photometric_loss, geometric_consistency_loss, smoothness_loss = compute_loss(images, depths, poses, updated_intrinsics)

            loss = photometric_loss + geometric_consistency_loss + smoothness_loss

            l2loss = args.l2_pose_reg_weight * l2_pose_regularization(poses) 
            loss += l2loss
            (loss / args.accumulate_steps).backward()

            
            if batch_idx % args.accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            run_wandb.log({
                "train/photometric_loss": photometric_loss.item(),
                "train/geometric_consistency_loss": geometric_consistency_loss.item(),
                "train/smoothness_loss": smoothness_loss.item(),
                "train/total_loss": loss.item(),
            })

            if batch_idx % 1000 == 0:
                log_images = []
                log_images.append(wandb.Image(
                    np.swapaxes(tensor2array(images[0][0]).T, 0, 1),
                    caption="val Input [0][0]"))
                log_images.append(wandb.Image(
                    np.swapaxes(tensor2array(depths[0][0], max_value=None, colormap='magma').T, 0, 1),
                    caption="val Dispnet Output Normalized [0][0]"))
                log_images.append(wandb.Image(
                    np.swapaxes(tensor2array(1/ (depths[0][0]), max_value=10, colormap='magma').T, 0, 1),
                    caption="val Depth Output [0][0]"))

                log_images.append(wandb.Image(
                    np.swapaxes(tensor2array(images[1][0]).T, 0, 1),
                    caption="val Input [1][0]"))
                log_images.append(wandb.Image(
                    np.swapaxes(tensor2array(depths[1][0], max_value=None, colormap='magma').T, 0, 1),
                    caption="val Dispnet Output Normalized [1][0]"))
                log_images.append(wandb.Image(
                    np.swapaxes(tensor2array(1/ (depths[1][0]), max_value=10, colormap='magma').T, 0, 1),
                    caption="val Depth Output [1][0]"))

                log_images.append(wandb.Image(
                    np.swapaxes(tensor2array(images[0][1]).T, 0, 1),
                    caption="val Input [0][1]"))
                log_images.append(wandb.Image(
                    np.swapaxes(tensor2array(depths[0][1], max_value=None, colormap='magma').T, 0, 1),
                    caption="val Dispnet Output Normalized [0][1]"))
                log_images.append(wandb.Image(
                    np.swapaxes(tensor2array(1/ (depths[0][1]), max_value=10, colormap='magma').T, 0, 1),
                    caption="val Depth Output [0][1]"))
        
                log_images.append(wandb.Image(
                    np.swapaxes(tensor2array(images[1][1]).T, 0, 1),
                    caption="val Input [1][1]"))
                log_images.append(wandb.Image(
                    np.swapaxes(tensor2array(depths[1][1], max_value=None, colormap='magma').T, 0, 1),
                    caption="val Dispnet Output Normalized [1][1]"))
                log_images.append(wandb.Image(
                    np.swapaxes(tensor2array(1/ (depths[1][1]), max_value=10, colormap='magma').T, 0, 1),
                    caption="val Depth Output [1][1]"))

                run_wandb.log({f"Train {batch_idx}": log_images, "epoch": epoch})
                import matplotlib.pyplot as plt
                plt.imsave(f"train_{batch_idx}.png", np.swapaxes(tensor2array(1/depths[0][0], max_value=10, colormap='magma').T, 0, 1))
            if batch_idx == 50000:
                break

            master_pbar.update(1)

        model.eval()
        with torch.no_grad():
            val_photometric_loss = []
            val_geometric_consistency_loss = []
            val_smoothness_loss = []
            for batch_idx, (images, images_individually_jittered, intrinsics) in tqdm(enumerate(val_loader)):
                images = [img.to(device) for img in images]
                intrinsics = intrinsics.to(device)

                depths, poses = model(images, intrinsics)  # model still gets the list

                images_t = torch.stack(images, dim=1)      # (B,T,3,H,W) for the loss
                photometric_loss, geometric_consistency_loss, smoothness_loss = compute_loss(
                    images_t, depths, poses, intrinsics
                )

                val_photometric_loss.append(photometric_loss.item())
                val_geometric_consistency_loss.append(geometric_consistency_loss.item())
                val_smoothness_loss.append(smoothness_loss.item())

                master_pbar.update(1)


            val_loss = np.mean(np.array(val_photometric_loss)+np.array(val_geometric_consistency_loss) + np.array(val_smoothness_loss))
            run_wandb.log({
                "val/photometric_loss": np.mean(val_photometric_loss),
                "val/geometric_consistency_loss": np.mean(val_geometric_consistency_loss),
                "val/smoothness_loss": np.mean(val_smoothness_loss),
                "val/total_loss": val_loss,
            })
            torch.save(model.state_dict(), args.name + "_last.pth")
            if val_loss < best_loss:
                print("BEST!")
                best_loss = val_loss
                torch.save(model.state_dict(), args.name + "_best.pth")
            if epoch % 10 == 0 and epoch > 0:
                torch.save(model.state_dict(), args.name + "_" + str(epoch) + ".pth")
if __name__ == '__main__':
    main()