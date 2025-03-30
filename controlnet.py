#!/usr/bin/env python
import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# === ControlNet3D Model ====================================================
# This is a simplified 3D UNet-like ControlNet that processes video (3D tensor) conditioning.
# It uses downsampling and upsampling blocks and outputs a residual signal (scaled by control_scale)
# that can be added to the LTXV latent predictions.
# (The model uses diffusers’ ModelMixin/ConfigMixin for compatibility.)
from diffusers.models.modeling_utils import ModelMixin, ConfigMixin, register_to_config

class ControlNet3D(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_down: int = 4,
        num_layers_per_block: int = 2,
        out_channels: int = None,
        control_scale: float = 1.0,
    ):
        """
        Args:
            in_channels: Number of input channels (e.g. RGB → 3).
            base_channels: The base number of channels used in the network.
            num_down: Number of downsampling blocks.
            out_channels: Number of output channels. If None, set to base_channels * (2**num_down).
            control_scale: A scaling factor for the control signal.
        """
        super().__init__()
        if out_channels is None:
            out_channels = base_channels * (2 ** num_down)
        self.control_scale = control_scale

        # Build downsampling blocks
        self.down_blocks = nn.ModuleList()
        channels = in_channels
        for i in range(num_down):
            block = nn.Sequential(
                nn.Conv3d(channels, base_channels * (2 ** i), kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(base_channels * (2 ** i), base_channels * (2 ** i), kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.down_blocks.append(block)
            channels = base_channels * (2 ** i)

        # Middle block
        self.mid_block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Build upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(num_down)):
            block = nn.Sequential(
                nn.Conv3d(channels, base_channels * (2 ** i), kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(base_channels * (2 ** i), base_channels * (2 ** i), kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.up_blocks.append(block)
            channels = base_channels * (2 ** i)
        self.out_conv = nn.Conv3d(channels, out_channels, kernel_size=1)

    def forward(self, control_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            control_input: Tensor of shape [B, C, F, H, W] representing the conditioning video.
        Returns:
            Residual control signal of shape [B, out_channels, F, H, W].
        """
        skips = []
        x = control_input
        # Downsampling path
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
            x = F.avg_pool3d(x, kernel_size=2)
        x = self.mid_block(x)
        # Upsampling path
        for block in self.up_blocks:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
            skip = skips.pop()
            # Match dimensions if necessary
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = x + skip
            x = block(x)
        x = self.out_conv(x)
        return x * self.config.control_scale

# === Video Dataset and Video Loader ========================================
# The VideoDataset loads pairs of videos from two directories.
# One directory contains the preprocessed (control) videos and the other the target (result) videos.
# Each video is loaded as a tensor of shape [C, F, H, W].
def load_video(video_path: str, num_frames: int = 16, resize: tuple = (64, 64)) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"Video at {video_path} has no frames.")
    # If the video has fewer frames than requested, loop it.
    if len(frames) < num_frames:
        frames = frames * (num_frames // len(frames) + 1)
    frames = frames[:num_frames]
    frames = np.stack(frames, axis=0)  # (F, H, W, C)
    frames = frames.transpose(3, 0, 1, 2)  # (C, F, H, W)
    frames = frames.astype(np.float32) / 255.0
    return torch.from_numpy(frames)

class VideoDataset(Dataset):
    def __init__(self, control_videos_dir: str, result_videos_dir: str, num_frames: int = 16, resize: tuple = (64, 64)):
        self.control_paths = sorted(glob.glob(str(Path(control_videos_dir) / "*.mp4")))
        self.result_paths = sorted(glob.glob(str(Path(result_videos_dir) / "*.mp4")))
        if len(self.control_paths) != len(self.result_paths):
            raise ValueError("Mismatch between number of control and result videos.")
        self.num_frames = num_frames
        self.resize = resize

    def __len__(self):
        return len(self.control_paths)

    def __getitem__(self, idx: int):
        control_video = load_video(self.control_paths[idx], num_frames=self.num_frames, resize=self.resize)
        result_video = load_video(self.result_paths[idx], num_frames=self.num_frames, resize=self.resize)
        return control_video, result_video

# === Training Script ========================================================
# In this simple training example we train ControlNet3D to predict the residual between the result video and
# the control (preprocessed) video.
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoDataset(args.control_dir, args.result_dir, num_frames=args.num_frames, resize=(args.resize_width, args.resize_height))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    control_net = ControlNet3D(
        in_channels=3,
        base_channels=args.base_channels,
        num_down=args.num_down,
        control_scale=args.control_scale,
    )
    control_net.to(device)

    optimizer = optim.Adam(control_net.parameters(), lr=args.learning_rate)
    mse_loss = nn.MSELoss()

    for epoch in range(args.num_epochs):
        control_net.train()
        epoch_loss = 0.0
        for i, (control_video, result_video) in enumerate(dataloader):
            control_video = control_video.to(device)  # shape: [B, C, F, H, W]
            result_video = result_video.to(device)
            optimizer.zero_grad()

            # Forward: predict residual = (result - control)
            control_out = control_net(control_video)
            target = result_video - control_video
            loss = mse_loss(control_out, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] Average Loss: {avg_loss:.4f}")

        os.makedirs(args.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.output_dir, f"controlnet_epoch_{epoch+1}.pt")
        torch.save(control_net.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

# === Inference Script =======================================================
# At inference time the trained ControlNet3D is applied to an input control video and its output is added
# to the control video to produce the final (guided) result.
def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    control_net = ControlNet3D(
        in_channels=3,
        base_channels=args.base_channels,
        num_down=args.num_down,
        control_scale=args.control_scale,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    control_net.load_state_dict(checkpoint)
    control_net.to(device)
    control_net.eval()

    control_video = load_video(args.input_video, num_frames=args.num_frames, resize=(args.resize_width, args.resize_height))
    control_video = control_video.unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        control_out = control_net(control_video)
    result_video = control_video + control_out  # add residual to control

    # Convert the tensor to a sequence of images and save to disk.
    result_video = result_video.squeeze(0).cpu()  # shape: [C, F, H, W]
    result_video = result_video.permute(1, 2, 3, 0).numpy()  # (F, H, W, C)
    result_video = (result_video * 255).astype(np.uint8)
    os.makedirs(args.output_video_dir, exist_ok=True)
    for i, frame in enumerate(result_video):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output_video_dir, f"frame_{i:04d}.png"), frame_bgr)
    print(f"Saved output frames to {args.output_video_dir}")

# === Main: Argument Parsing and Mode Switching =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or infer ControlNet3D for LTXV")
    parser.add_argument("--mode", type=str, choices=["train", "infer"], required=True, help="Mode: train or infer")
    parser.add_argument("--control_dir", type=str, default="./control_videos", help="Directory with control (preprocessed) videos")
    parser.add_argument("--result_dir", type=str, default="./result_videos", help="Directory with result videos (training targets)")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames per video")
    parser.add_argument("--resize_width", type=int, default=64, help="Width to resize video frames")
    parser.add_argument("--resize_height", type=int, default=64, help="Height to resize video frames")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channels for ControlNet")
    parser.add_argument("--num_down", type=int, default=4, help="Number of downsampling blocks")
    parser.add_argument("--control_scale", type=float, default=1.0, help="Scaling factor for the control signal")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval (in steps) for logging during training")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    # Inference options
    parser.add_argument("--checkpoint", type=str, help="Path to a trained checkpoint (for inference)")
    parser.add_argument("--input_video", type=str, help="Path to an input control video for inference")
    parser.add_argument("--output_video_dir", type=str, default="./output_frames", help="Directory to save inference output frames")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
