"""Train and export a grayscale pupil-localisation model on LPW.

The script consumes the original LPW layout (participant/video.avi + video.txt),
builds a memory-mapped cache once, splits by participant, trains a lightweight
spatial-softmax model, and exports an opset-11 ONNX model with output [nx, ny].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Config:
    image_size: int = 224
    frame_stride: int = 3
    batch_size: int = 64
    epochs: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.18
    seed: int = 42
    workers: int = 4
    patience: int = 8


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def discover_videos(root: Path) -> list[tuple[str, Path, Path]]:
    records: list[tuple[str, Path, Path]] = []
    for video in sorted(root.glob("*/*.avi"), key=lambda p: (int(p.parent.name), int(p.stem))):
        label = video.with_suffix(".txt")
        if label.exists():
            records.append((video.parent.name, video, label))
    if not records:
        raise FileNotFoundError(f"No LPW participant/video.avi + video.txt pairs found in {root}")
    return records


def read_points(path: Path) -> np.ndarray:
    points = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if points.shape[1] != 2:
        raise ValueError(f"Expected x y labels in {path}, got shape {points.shape}")
    return points


def build_cache(lpw_root: Path, cache_dir: Path, cfg: Config, rebuild: bool = False) -> Path:
    """Decode videos sequentially into a compact uint8 memmap cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "metadata.json"
    images_path = cache_dir / "images.uint8"
    labels_path = cache_dir / "labels.npy"
    samples_path = cache_dir / "samples.csv"
    videos = discover_videos(lpw_root)
    signature = {
        "image_size": cfg.image_size,
        "frame_stride": cfg.frame_stride,
        "videos": [[str(v.relative_to(lpw_root)), v.stat().st_size, l.stat().st_size]
                   for _, v, l in videos],
    }
    if not rebuild and all(p.exists() for p in (meta_path, images_path, labels_path, samples_path)):
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_bytes = old.get("samples", 0) * cfg.image_size * cfg.image_size
        if old.get("signature") == signature and images_path.stat().st_size == expected_bytes:
            print(f"Using cache: {cache_dir} ({old['samples']} samples)")
            return cache_dir

    counts = []
    for participant, video, label in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        label_count = len(read_points(label))
        usable = min(frame_count, label_count)
        counts.append((participant, video, label, usable, math.ceil(usable / cfg.frame_stride)))
    total = sum(item[4] for item in counts)
    if total == 0:
        raise RuntimeError("LPW cache would contain zero samples")

    tmp_images = images_path.with_suffix(".tmp")
    mm = np.memmap(tmp_images, mode="w+", dtype=np.uint8,
                   shape=(total, cfg.image_size, cfg.image_size))
    all_labels = np.empty((total, 2), dtype=np.float32)
    rows: list[tuple[str, str, int]] = []
    cursor = 0
    started = time.time()
    for video_idx, (participant, video, label, usable, _) in enumerate(counts, 1):
        points = read_points(label)
        cap = cv2.VideoCapture(str(video))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width < 2 or height < 2:
            raise RuntimeError(f"Invalid video dimensions for {video}")
        frame_idx = 0
        while frame_idx < usable:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % cfg.frame_stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                mm[cursor] = cv2.resize(gray, (cfg.image_size, cfg.image_size),
                                        interpolation=cv2.INTER_AREA)
                x, y = points[frame_idx]
                all_labels[cursor] = (np.clip(x / (width - 1), 0.0, 1.0),
                                      np.clip(y / (height - 1), 0.0, 1.0))
                rows.append((participant, video.stem, frame_idx))
                cursor += 1
            frame_idx += 1
        cap.release()
        print(f"Cache {video_idx:02d}/{len(counts)}: {video.relative_to(lpw_root)}")
    mm.flush()
    del mm
    if cursor != total:
        # Rare truncated/corrupt videos: shrink the cache without loading it all.
        with open(tmp_images, "r+b") as handle:
            handle.truncate(cursor * cfg.image_size * cfg.image_size)
        all_labels = all_labels[:cursor]
    os.replace(tmp_images, images_path)
    np.save(labels_path, all_labels)
    with samples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("participant", "video", "frame"))
        writer.writerows(rows)
    meta_path.write_text(json.dumps({"signature": signature, "samples": cursor}, indent=2),
                         encoding="utf-8")
    print(f"Built cache: {cursor} samples in {time.time() - started:.1f}s")
    return cache_dir


def participant_split(samples_csv: Path, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    participants = []
    with samples_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            participants.append(row["participant"])
    unique = sorted(set(participants), key=int)
    rng = random.Random(seed)
    rng.shuffle(unique)
    val_count = max(1, round(len(unique) * val_fraction))
    val_people = set(unique[:val_count])
    train_idx = np.asarray([i for i, p in enumerate(participants) if p not in val_people], dtype=np.int64)
    val_idx = np.asarray([i for i, p in enumerate(participants) if p in val_people], dtype=np.int64)
    return train_idx, val_idx, sorted(val_people, key=int)


class LPWCacheDataset(Dataset):
    def __init__(self, cache_dir: Path, indices: Iterable[int], image_size: int,
                 augment: bool = False):
        self.cache_dir = cache_dir
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.image_size = image_size
        self.augment = augment
        self.labels = np.load(cache_dir / "labels.npy", mmap_mode="r")
        self._images = None

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def images(self):
        if self._images is None:  # Open independently inside each DataLoader worker.
            count = (self.cache_dir / "images.uint8").stat().st_size // (self.image_size ** 2)
            self._images = np.memmap(self.cache_dir / "images.uint8", mode="r", dtype=np.uint8,
                                     shape=(count, self.image_size, self.image_size))
        return self._images

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        image = np.asarray(self.images[idx]).copy()
        label = np.asarray(self.labels[idx], dtype=np.float32).copy()
        if self.augment:
            image, label = self._augment(image, label)
        tensor = torch.from_numpy(image).unsqueeze(0).float().div_(255.0)
        return tensor, torch.from_numpy(label)

    @staticmethod
    def _augment(image: np.ndarray, label: np.ndarray):
        h, w = image.shape
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            label[0] = 1.0 - label[0]
        # Small translation teaches coordinate equivariance while keeping the pupil visible.
        if random.random() < 0.6:
            dx = random.uniform(-0.06, 0.06) * w
            dy = random.uniform(-0.06, 0.06) * h
            moved = label + np.asarray([dx / (w - 1), dy / (h - 1)], dtype=np.float32)
            if np.all((moved > 0.02) & (moved < 0.98)):
                image = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                label = moved
        alpha = random.uniform(0.72, 1.28)
        beta = random.uniform(-24.0, 24.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255)
        if random.random() < 0.25:
            image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.2, 1.2))
        if random.random() < 0.25:
            image += np.random.normal(0, random.uniform(1, 7), image.shape)
        if random.random() < 0.2:  # Reflection/eyelash-like occlusion away from the target.
            ow, oh = random.randint(w // 20, w // 6), random.randint(h // 30, h // 10)
            ox, oy = random.randrange(max(1, w - ow)), random.randrange(max(1, h - oh))
            px, py = label * np.asarray([w - 1, h - 1])
            if not (ox - 5 <= px <= ox + ow + 5 and oy - 5 <= py <= oy + oh + 5):
                image[oy:oy + oh, ox:ox + ow] = random.uniform(0, 255)
        return np.clip(image, 0, 255).astype(np.uint8), label.astype(np.float32)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )


class PupilHeatmapNet(nn.Module):
    """Lightweight heatmap + differentiable expectation coordinate head."""
    def __init__(self, image_size: int = 224):
        super().__init__()
        if image_size % 16:
            raise ValueError("image_size must be divisible by 16")
        self.features = nn.Sequential(
            ConvBlock(1, 16, 2), ConvBlock(16, 16),
            ConvBlock(16, 24, 2), ConvBlock(24, 24),
            ConvBlock(24, 32, 2), ConvBlock(32, 32),
            ConvBlock(32, 48, 2), ConvBlock(48, 48),
        )
        self.heatmap = nn.Conv2d(48, 1, 1)
        side = image_size // 16
        yy, xx = torch.meshgrid(torch.linspace(0, 1, side), torch.linspace(0, 1, side), indexing="ij")
        self.register_buffer("x_grid", xx.reshape(1, -1), persistent=True)
        self.register_buffer("y_grid", yy.reshape(1, -1), persistent=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.heatmap(self.features(image)).flatten(1)
        probability = torch.softmax(logits, dim=1)
        x = torch.sum(probability * self.x_grid, dim=1)
        y = torch.sum(probability * self.y_grid, dim=1)
        return torch.stack((x, y), dim=1)


def pixel_errors(prediction: torch.Tensor, target: torch.Tensor, image_size: int) -> torch.Tensor:
    return torch.linalg.vector_norm((prediction - target) * (image_size - 1), dim=1)


@torch.inference_mode()
def evaluate(model, loader, device, image_size):
    model.eval()
    errors = []
    loss_sum = 0.0
    criterion = nn.SmoothL1Loss(beta=0.02, reduction="sum")
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss_sum += criterion(outputs, labels).item()
        errors.append(pixel_errors(outputs, labels, image_size).cpu())
    error = torch.cat(errors)
    return loss_sum / len(loader.dataset), error.mean().item(), torch.quantile(error, 0.95).item()


def export_onnx(model: nn.Module, path: Path, image_size: int, device: torch.device) -> None:
    model.eval()
    dummy = torch.rand(1, 1, image_size, image_size, device=device)
    torch.onnx.export(model, dummy, path, input_names=["gray"], output_names=["pupil_xy"],
                      opset_version=11, do_constant_folding=True,
                      dynamo=False,
                      dynamic_axes={"gray": {0: "batch"}, "pupil_xy": {0: "batch"}})
    print(f"Exported ONNX: {path}")


def verify_onnx(model: nn.Module, path: Path, loader: DataLoader, device: torch.device) -> None:
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    images, _ = next(iter(loader))
    images = images[: min(8, len(images))]
    with torch.inference_mode():
        torch_output = model(images.to(device)).cpu().numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_output = session.run(["pupil_xy"], {"gray": images.numpy()})[0]
    max_delta = float(np.max(np.abs(torch_output - ort_output)))
    if max_delta > 1e-4:
        raise RuntimeError(f"ONNX verification failed: max |PyTorch-ONNX| = {max_delta:.6g}")
    print(f"ONNX verified: input {images.shape}, output {ort_output.shape}, max delta {max_delta:.3g}")


def train(args) -> None:
    cfg = Config(image_size=args.image_size, frame_stride=args.frame_stride,
                 batch_size=args.batch_size, epochs=args.epochs,
                 learning_rate=args.learning_rate, workers=args.workers, seed=args.seed)
    seed_everything(cfg.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = build_cache(args.lpw.resolve(), args.cache.resolve(), cfg, args.rebuild_cache)
    train_idx, val_idx, val_people = participant_split(cache / "samples.csv", cfg.val_fraction, cfg.seed)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, val participants={val_people}")
    (output / "config.json").write_text(json.dumps({**asdict(cfg), "lpw": str(args.lpw.resolve()),
                                                     "val_participants": val_people}, indent=2),
                                                encoding="utf-8")

    pin = torch.cuda.is_available()
    loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=pin,
                       persistent_workers=cfg.workers > 0)
    train_loader = DataLoader(LPWCacheDataset(cache, train_idx, cfg.image_size, True),
                              shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(LPWCacheDataset(cache, val_idx, cfg.image_size, False),
                            shuffle=False, **loader_args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = PupilHeatmapNet(cfg.image_size).to(device)
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs,
                                                            eta_min=cfg.learning_rate * 0.03)
    criterion = nn.SmoothL1Loss(beta=0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_error, stale = float("inf"), 0
    history = []
    best_path = output / "pupil_lpw_best.pth"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        started = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(images)
        scheduler.step()
        val_loss, mean_px, p95_px = evaluate(model, val_loader, device, cfg.image_size)
        row = {"epoch": epoch, "train_loss": train_loss / len(train_idx), "val_loss": val_loss,
               "mean_error_px": mean_px, "p95_error_px": p95_px,
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - started}
        history.append(row)
        print(f"Epoch {epoch:02d}/{cfg.epochs} train={row['train_loss']:.5f} val={val_loss:.5f} "
              f"mean={mean_px:.2f}px p95={p95_px:.2f}px time={row['seconds']:.1f}s")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if mean_px < best_error:
            best_error, stale = mean_px, 0
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "metrics": row}, best_path)
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"Early stopping after {stale} epochs without improvement")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    onnx_path = output / "pupil_lpw_opset11.onnx"
    export_onnx(model, onnx_path, cfg.image_size, device)
    verify_onnx(model, onnx_path, val_loader, device)
    print(f"Best validation mean error: {checkpoint['metrics']['mean_error_px']:.2f}px")


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lpw", type=Path, default=here.parent.parent / "LPW")
    parser.add_argument("--cache", type=Path, default=here / ".lpw_cache")
    parser.add_argument("--output", type=Path, default=here / "outputs")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frame-stride", type=int, default=3,
                        help="Use every Nth frame; 3 gives ~22k samples")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
