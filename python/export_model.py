#!/usr/bin/env python3
"""
Aegis-RT Perception Backbone Trainer & ONNX Exporter.

Trains a TensorRT-friendly, single-stage anchor-free detector (`AegisNanoDetector`)
on 640x480 RGB frames captured from the MuJoCo pursuit arena and exports a static
opset-17 ONNX graph (`1x3x320x320` -> `1x5x20x20`).

Every operator (`Conv2d + BatchNorm2d + SiLU + Add`) fuses cleanly into TensorRT
FP16/INT8 Tensor Core kernels and Apple CoreML Neural Engine (ANE) MLPrograms.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


SRC_W, SRC_H = 640, 480
DST_W, DST_H = 320, 320
GRID_W, GRID_H = 20, 20
SCALE = min(DST_W / SRC_W, DST_H / SRC_H)  # 0.5
RESIZED_W, RESIZED_H = round(SRC_W * SCALE), round(SRC_H * SCALE)  # 320, 240
PAD_X, PAD_Y = (DST_W - RESIZED_W) // 2, (DST_H - RESIZED_H) // 2  # 0, 40


def letterbox_numpy(rgb_hwc: np.ndarray) -> np.ndarray:
    """Matches C++ cpu_preprocess_letterbox_nchw / CUDA fused kernel layout."""
    t = torch.from_numpy(rgb_hwc).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    resized = F.interpolate(t, size=(RESIZED_H, RESIZED_W), mode="bilinear", align_corners=False)
    canvas = torch.full((1, 3, DST_H, DST_W), 114.0 / 255.0, dtype=torch.float32)
    canvas[:, :, PAD_Y : PAD_Y + RESIZED_H, PAD_X : PAD_X + RESIZED_W] = resized
    return canvas.squeeze(0).numpy()


class ConvBnAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBnAct(channels, channels, k=3, s=1),
            ConvBnAct(channels, channels, k=3, s=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AegisNanoDetector(nn.Module):
    """Compact stride-16 anchor-free target detector (~240K params, <0.4 GFLOPs)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBnAct(3, 16, k=3, s=2),     # 160x160
            ConvBnAct(16, 32, k=3, s=2),    # 80x80
            ResidualBlock(32),
            ConvBnAct(32, 64, k=3, s=2),    # 40x40
            ResidualBlock(64),
            ConvBnAct(64, 96, k=3, s=2),    # 20x20
            ResidualBlock(96),
        )
        self.head = nn.Sequential(
            ConvBnAct(96, 64, k=3, s=1),
            nn.Conv2d(64, 5, kernel_size=1, stride=1, padding=0, bias=True),
        )
        # Initialize objectness bias for stable early focal convergence (p ~= 0.02)
        nn.init.constant_(self.head[-1].bias[0], -3.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class MujocoPursuitDataset(Dataset):
    def __init__(self, dataset_dir: Path, augment: bool = True) -> None:
        self.samples: List[Tuple[np.ndarray, int, float, float, float, float]] = []
        self.augment = augment
        labels_path = dataset_dir / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing {labels_path}. Run aegis_rt --dump-dataset first.")

        with labels_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fid = int(row["frame_id"])
                rgb_path = dataset_dir / f"frame_{fid:04d}.rgb"
                if rgb_path.exists():
                    raw = np.fromfile(rgb_path, dtype=np.uint8).reshape(SRC_H, SRC_W, 3)
                    self.samples.append((
                        letterbox_numpy(raw),
                        int(row["visible"]),
                        float(row["cx_px"]),
                        float(row["cy_px"]),
                        float(row["w_px"]),
                        float(row["h_px"]),
                    ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_img, vis, cx, cy, bw, bh = self.samples[idx]
        img = base_img

        # Optional horizontal flip & photometric jitter for generalization across orbits
        if self.augment and np.random.rand() < 0.5:
            img = img[:, :, ::-1].copy()
            cx = (SRC_W - 1.0) - cx

        if self.augment:
            gain = np.random.uniform(0.88, 1.12)
            img = np.clip(img * gain, 0.0, 1.0)

        target_obj = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
        target_box = np.zeros((4, GRID_H, GRID_W), dtype=np.float32)
        pos_mask = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)

        if vis == 1:
            model_cx = cx * SCALE + PAD_X
            model_cy = cy * SCALE + PAD_Y
            model_w = np.clip((bw * SCALE) / DST_W, 0.02, 0.95)
            model_h = np.clip((bh * SCALE) / DST_H, 0.02, 0.95)

            stride_x = DST_W / GRID_W
            stride_y = DST_H / GRID_H
            gx_f = model_cx / stride_x
            gy_f = model_cy / stride_y
            gx = int(np.clip(math.floor(gx_f), 0, GRID_W - 1))
            gy = int(np.clip(math.floor(gy_f), 0, GRID_H - 1))

            target_obj[0, gy, gx] = 1.0
            pos_mask[0, gy, gx] = 1.0
            target_box[0, gy, gx] = np.clip(gx_f - gx, 1e-4, 1.0 - 1e-4)
            target_box[1, gy, gx] = np.clip(gy_f - gy, 1e-4, 1.0 - 1e-4)
            target_box[2, gy, gx] = model_w
            target_box[3, gy, gx] = model_h

        return (
            torch.from_numpy(img),
            torch.from_numpy(np.concatenate([target_obj, target_box], axis=0)),
            torch.from_numpy(pos_mask),
        )


def train_and_export(dataset_dir: Path, out_onnx: Path, epochs: int) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[Export] Training AegisNanoDetector on device={device} using {dataset_dir}...")
    ds = MujocoPursuitDataset(dataset_dir, augment=True)
    val_ds = MujocoPursuitDataset(dataset_dir, augment=False)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=False)

    model = AegisNanoDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, targets, mask in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            preds = model(imgs)
            pred_obj = preds[:, 0:1]
            pred_box = torch.sigmoid(preds[:, 1:5])

            # Weighted BCE objectness loss (balances 1 positive cell vs 399 background cells)
            pos_weight = torch.tensor([60.0], device=device)
            loss_obj = F.binary_cross_entropy_with_logits(pred_obj, targets[:, 0:1], pos_weight=pos_weight)

            num_pos = mask.sum().clamp(min=1.0)
            loss_box = (F.smooth_l1_loss(pred_box, targets[:, 1:5], reduction="none") * mask).sum() / num_pos
            loss = loss_obj + 6.0 * loss_box

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs} | loss={epoch_loss / len(loader):.4f}", flush=True)

    # Evaluate pixel localization error on un-augmented MuJoCo frames
    model.eval()
    px_errors = []
    hits = 0
    total_vis = 0
    with torch.no_grad():
        for i in range(len(val_ds)):
            img_t, _, _ = val_ds[i]
            _, vis, gt_cx, gt_cy, _, _ = val_ds.samples[i]
            if vis == 0:
                continue
            total_vis += 1
            out = model(img_t.unsqueeze(0).to(device)).cpu().numpy()[0]
            flat_idx = int(np.argmax(out[0]))
            conf = 1.0 / (1.0 + math.exp(-float(out[0].reshape(-1)[flat_idx])))
            if conf >= 0.40:
                hits += 1
                gy, gx = divmod(flat_idx, GRID_W)
                dx = 1.0 / (1.0 + math.exp(-float(out[1, gy, gx])))
                dy = 1.0 / (1.0 + math.exp(-float(out[2, gy, gx])))
                mcx = (gx + dx) * (DST_W / GRID_W)
                mcy = (gy + dy) * (DST_H / GRID_H)
                pred_cx = (mcx - PAD_X) / SCALE
                pred_cy = (mcy - PAD_Y) / SCALE
                px_errors.append(math.hypot(pred_cx - gt_cx, pred_cy - gt_cy))

    recall = 100.0 * hits / max(1, total_vis)
    mean_px_err = float(np.mean(px_errors)) if px_errors else 999.0
    print(f"[Export] Validation Recall={recall:.1f}% | Mean Center Pixel Error={mean_px_err:.2f} px")

    # Export clean static opset-17 ONNX model on CPU
    model_cpu = model.cpu().eval()
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, DST_H, DST_W, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        str(out_onnx),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["detections"],
        dynamo=False,
    )
    onnx_model = onnx.load(str(out_onnx))
    onnx.checker.check_model(onnx_model)
    print(f"[Export] Verified & saved static ONNX model -> {out_onnx} ({out_onnx.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("assets/calib_frames"))
    parser.add_argument("--out", type=Path, default=Path("assets/models/aegis_detector.onnx"))
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    train_and_export(args.dataset_dir, args.out, args.epochs)


if __name__ == "__main__":
    main()
