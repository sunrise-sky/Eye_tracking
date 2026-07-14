"""Train and export an A1-compatible pupil centre regression model.

The input datasets are expected at ``p1-left`` and ``p1-right``.  Each dataset
contains ``frames/<id>-eye.png`` and a sparse ``pupil-ellipses.txt`` file with
rows of the form::

    frame_id | center_x center_y axis_a axis_b angle

Only the ellipse centre is used.  Image resizing and normalisation happen on
the CPU, exactly as in the embedded demo.  The exported graph itself contains
only Conv, Relu and MaxPool operators; coordinate clipping stays on the CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ANNOTATION_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$"
)
A1_ALLOWED_OPS = {
    "Conv", "AveragePool", "GlobalAveragePool", "MaxPool",
    "BatchNormalization", "Add", "Mul", "Concat", "Split", "Relu",
    "LeakyRelu", "Transpose", "Resize", "ConvTranspose", "Upsample",
}


@dataclass(frozen=True)
class Config:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 120
    learning_rate: float = 1e-4
    backbone_lr_scale: float = 0.25
    weight_decay: float = 2e-4
    val_fraction: float = 0.20
    temporal_block: int = 100
    seed: int = 42
    workers: int = 0
    patience: int = 25
    freeze_backbone_epochs: int = 3


@dataclass(frozen=True)
class Sample:
    side: str
    frame_id: int
    image: Path
    x: float
    y: float
    width: int
    height: int

    @property
    def normalized_xy(self) -> tuple[float, float]:
        return self.x / (self.width - 1), self.y / (self.height - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Benchmark mode is useful for the fixed 224x224 input.  Seeding keeps the
    # data split stable; exact bit reproducibility is not required for training.
    torch.backends.cudnn.benchmark = True


def imread_gray(path: Path) -> np.ndarray | None:
    """Read Unicode Windows paths, which cv2.imread does not handle reliably."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def read_dataset(root: Path, side: str) -> list[Sample]:
    annotations = root / "pupil-ellipses.txt"
    frames = root / "frames"
    if not annotations.is_file() or not frames.is_dir():
        raise FileNotFoundError(
            f"Expected {annotations} and {frames}; dataset layout is incomplete"
        )

    samples: list[Sample] = []
    dimensions: tuple[int, int] | None = None
    for line_number, line in enumerate(
        annotations.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        match = ANNOTATION_RE.match(line)
        if not match:
            raise ValueError(f"Malformed annotation {annotations}:{line_number}: {line!r}")
        frame_id = int(match.group(1))
        x, y = float(match.group(2)), float(match.group(3))
        image = frames / f"{frame_id}-eye.png"
        if not image.is_file():
            raise FileNotFoundError(f"Annotated frame is missing: {image}")
        decoded = imread_gray(image)
        if decoded is None:
            raise ValueError(f"Cannot decode image: {image}")
        height, width = decoded.shape
        if dimensions is None:
            dimensions = (width, height)
        elif dimensions != (width, height):
            raise ValueError(
                f"Mixed image sizes are not supported: {image} is {width}x{height}, "
                f"expected {dimensions[0]}x{dimensions[1]}"
            )
        if not (0.0 <= x < width and 0.0 <= y < height):
            raise ValueError(
                f"Pupil centre outside image at {annotations}:{line_number}: "
                f"({x}, {y}) not in {width}x{height}"
            )
        samples.append(Sample(side, frame_id, image, x, y, width, height))

    if not samples:
        raise ValueError(f"No annotations found in {annotations}")
    return samples


def temporal_group_split(
    samples: list[Sample], val_fraction: float, block_size: int, seed: int
) -> tuple[list[Sample], list[Sample], list[int]]:
    """Split common left/right time blocks so adjacent frames do not leak."""
    if block_size < 1:
        raise ValueError("temporal_block must be positive")
    groups = sorted({sample.frame_id // block_size for sample in samples})
    if len(groups) < 2:
        raise ValueError("Need at least two temporal blocks for train/validation split")
    shuffled = groups.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = min(len(groups) - 1, max(1, round(len(groups) * val_fraction)))
    val_groups = set(shuffled[:val_count])
    train = [s for s in samples if s.frame_id // block_size not in val_groups]
    val = [s for s in samples if s.frame_id // block_size in val_groups]
    if not train or not val:
        raise RuntimeError("Temporal split produced an empty partition")
    for side in {s.side for s in samples}:
        if not any(s.side == side for s in train) or not any(s.side == side for s in val):
            raise RuntimeError(f"Temporal split does not contain {side} in both partitions")
    return train, val, sorted(val_groups)


class PupilDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, augment: bool) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = imread_gray(sample.image)
        if image is None:
            raise RuntimeError(f"Cannot decode image during training: {sample.image}")
        label = np.asarray(sample.normalized_xy, dtype=np.float32)
        if self.augment:
            image, label = self._augment(image, label)
        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float()
        tensor.mul_(1.0 / 255.0)
        return tensor, torch.from_numpy(label), sample.side

    @staticmethod
    def _augment(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            label[0] = 1.0 - label[0]

        # Affine perturbation improves robustness to imperfect eye ROI boxes.
        if random.random() < 0.85:
            angle = random.uniform(-5.0, 5.0)
            scale = random.uniform(0.92, 1.08)
            dx = random.uniform(-0.055, 0.055) * width
            dy = random.uniform(-0.055, 0.055) * height
            matrix = cv2.getRotationMatrix2D(
                ((width - 1) * 0.5, (height - 1) * 0.5), angle, scale
            ).astype(np.float32)
            matrix[:, 2] += (dx, dy)
            point = np.asarray(
                [label[0] * (width - 1), label[1] * (height - 1), 1.0],
                dtype=np.float32,
            )
            moved = matrix @ point
            moved /= np.asarray([width - 1, height - 1], dtype=np.float32)
            if np.all((moved > 0.015) & (moved < 0.985)):
                image = cv2.warpAffine(
                    image,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                label = moved.astype(np.float32)

        image_f = image.astype(np.float32)
        if random.random() < 0.9:
            image_f = image_f * random.uniform(0.72, 1.28) + random.uniform(-22.0, 22.0)
        if random.random() < 0.25:
            image_f = cv2.GaussianBlur(image_f, (3, 3), random.uniform(0.15, 1.0))
        if random.random() < 0.25:
            image_f += np.random.normal(0.0, random.uniform(1.0, 6.0), image_f.shape)

        # Small occlusions mimic reflections and eyelashes without covering the label.
        if random.random() < 0.18:
            ow = random.randint(max(2, width // 30), max(3, width // 10))
            oh = random.randint(max(2, height // 40), max(3, height // 12))
            ox = random.randrange(max(1, width - ow))
            oy = random.randrange(max(1, height - oh))
            px, py = label * np.asarray([width - 1, height - 1])
            if not (ox - 8 <= px <= ox + ow + 8 and oy - 8 <= py <= oy + oh + 8):
                image_f[oy : oy + oh, ox : ox + ow] = random.uniform(0.0, 255.0)

        return np.clip(image_f, 0, 255).astype(np.uint8), label.astype(np.float32)


class PupilA1Net(nn.Module):
    """Spatial coordinate regressor using A1-native operators only.

    A final full-field convolution replaces Flatten/Gemm.  Unlike global average
    pooling, it keeps a separate learned weight for each spatial feature cell.
    """

    def __init__(self, image_size: int = 224) -> None:
        super().__init__()
        if image_size % 32:
            raise ValueError("image_size must be divisible by 32")
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.spatial_projection = nn.Conv2d(64, 16, 1)
        feature_side = image_size // 32
        self.coordinate_head = nn.Conv2d(16, 2, feature_side, bias=True)
        nn.init.zeros_(self.coordinate_head.weight)
        nn.init.constant_(self.coordinate_head.bias, 0.4)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.pool(self.relu(self.conv1(image)))
        image = self.pool(self.relu(self.conv2(image)))
        image = self.pool(self.relu(self.conv3(image)))
        image = self.pool(self.relu(self.conv4(image)))
        image = self.pool(image)
        image = self.relu(self.spatial_projection(image))
        return self.coordinate_head(image)  # [N, 2, 1, 1], raw normalized xy

    def backbone_parameters(self):
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4):
            yield from layer.parameters()

    def head_parameters(self):
        yield from self.spatial_projection.parameters()
        yield from self.coordinate_head.parameters()


def load_pretrained_backbone(model: PupilA1Net, checkpoint: Path | None) -> bool:
    if checkpoint is None or not checkpoint.is_file():
        return False
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    loaded = []
    for name in ("conv1", "conv2", "conv3", "conv4"):
        layer = getattr(model, name)
        weight, bias = state.get(f"{name}.weight"), state.get(f"{name}.bias")
        if weight is None or bias is None:
            raise ValueError(f"Pretrained checkpoint is missing {name} weights")
        if layer.weight.shape != weight.shape or layer.bias.shape != bias.shape:
            raise ValueError(f"Pretrained {name} shape does not match the A1 model")
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)
        loaded.append(name)
    print(f"Loaded pretrained backbone from {checkpoint}: {', '.join(loaded)}")
    return True


def xy_view(output: torch.Tensor) -> torch.Tensor:
    """Flatten outside the model so Reshape is never exported to ONNX."""
    return output[:, :, 0, 0]


def sample_errors(
    prediction: torch.Tensor, target: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    scale = prediction.new_tensor([width - 1, height - 1])
    delta = (prediction - target) * scale
    return torch.linalg.vector_norm(delta, dim=1)


@torch.inference_mode()
def evaluate(model, loader, device, width: int, height: int) -> dict[str, float]:
    model.eval()
    criterion = nn.SmoothL1Loss(beta=0.025, reduction="sum")
    loss_sum = 0.0
    all_errors: list[torch.Tensor] = []
    side_errors: dict[str, list[torch.Tensor]] = {"left": [], "right": []}
    for images, labels, sides in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        prediction = xy_view(model(images))
        loss_sum += criterion(prediction, labels).item()
        errors = sample_errors(prediction, labels, width, height).cpu()
        all_errors.append(errors)
        for i, side in enumerate(sides):
            side_errors[side].append(errors[i : i + 1])
    error = torch.cat(all_errors)
    metrics = {
        "val_loss": loss_sum / len(loader.dataset),
        "mean_error_px": error.mean().item(),
        "median_error_px": error.median().item(),
        "p95_error_px": torch.quantile(error, 0.95).item(),
    }
    for side, values in side_errors.items():
        if values:
            metrics[f"{side}_mean_error_px"] = torch.cat(values).mean().item()
    return metrics


def export_and_verify(
    model: nn.Module, path: Path, image_size: int, loader: DataLoader, device: torch.device
) -> dict[str, object]:
    import onnx
    import onnxruntime as ort

    model.eval()
    dummy = torch.rand(1, 1, image_size, image_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["gray"],
        output_names=["pupil_xy_raw"],
        opset_version=11,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(str(path))
    # Opset 11 models use IR v6.  Pinning this avoids newer-IR parser issues in
    # conservative embedded toolchains without changing graph semantics.
    graph.ir_version = 6
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))

    op_counts: dict[str, int] = {}
    unsupported: set[str] = set()
    for node in graph.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        if node.op_type not in A1_ALLOWED_OPS:
            unsupported.add(node.op_type)
        if node.op_type == "MaxPool":
            attrs = {attribute.name: attribute for attribute in node.attribute}
            kernel = list(attrs["kernel_shape"].ints)
            if any(value > 8 for value in kernel):
                raise RuntimeError(f"A1 MaxPool limit exceeded: kernel={kernel}")
        if node.op_type == "LeakyRelu":
            attrs = {attribute.name: attribute for attribute in node.attribute}
            alpha = attrs.get("alpha")
            value = alpha.f if alpha is not None else 0.01
            if not math.isclose(value, 0.1) and not math.isclose(value, 0.01):
                raise RuntimeError(f"A1-incompatible LeakyRelu alpha={value}")
    if unsupported:
        raise RuntimeError(f"A1-incompatible ONNX operators: {sorted(unsupported)}")

    images, _, _ = next(iter(loader))
    image = images[:1].numpy()
    with torch.inference_mode():
        torch_output = model(torch.from_numpy(image).to(device)).cpu().numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_output = session.run(["pupil_xy_raw"], {"gray": image})[0]
    max_delta = float(np.max(np.abs(torch_output - ort_output)))
    if max_delta > 1e-4:
        raise RuntimeError(f"ONNX numerical check failed: max delta={max_delta:.6g}")
    details = {
        "path": str(path.resolve()),
        "opset": 11,
        "ir_version": graph.ir_version,
        "input": {"name": "gray", "shape": [1, 1, image_size, image_size], "dtype": "float32"},
        "output": {"name": "pupil_xy_raw", "shape": [1, 2, 1, 1], "dtype": "float32"},
        "operator_counts": op_counts,
        "unsupported_operators": sorted(unsupported),
        "pytorch_onnx_max_abs_delta": max_delta,
        "size_bytes": path.stat().st_size,
    }
    print(
        f"ONNX verified: ops={op_counts}, max_delta={max_delta:.3g}, "
        f"size={path.stat().st_size / 1024:.1f} KiB"
    )
    return details


def save_split(path: Path, train: list[Sample], val: list[Sample], groups: list[int]) -> None:
    content = {
        "strategy": "shared left/right temporal block split",
        "validation_blocks": groups,
        "train": [f"{s.side}:{s.frame_id}" for s in train],
        "validation": [f"{s.side}:{s.frame_id}" for s in val],
    }
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    cfg = Config(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        temporal_block=args.temporal_block,
        seed=args.seed,
        workers=args.workers,
        patience=args.patience,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
    )
    seed_everything(cfg.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    samples = read_dataset(args.left.resolve(), "left") + read_dataset(
        args.right.resolve(), "right"
    )
    train_samples, val_samples, val_groups = temporal_group_split(
        samples, cfg.val_fraction, cfg.temporal_block, cfg.seed
    )
    width, height = samples[0].width, samples[0].height
    print(
        f"Dataset: {len(samples)} labels ({sum(s.side == 'left' for s in samples)} left, "
        f"{sum(s.side == 'right' for s in samples)} right), source={width}x{height}"
    )
    print(
        f"Split: train={len(train_samples)}, val={len(val_samples)}, "
        f"validation temporal blocks={val_groups}"
    )
    save_split(output / "split.json", train_samples, val_samples, val_groups)

    pin = torch.cuda.is_available() and not args.cpu
    common_loader = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.workers,
        "pin_memory": pin,
    }
    if cfg.workers > 0:
        common_loader["persistent_workers"] = True
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        PupilDataset(train_samples, cfg.image_size, augment=True),
        shuffle=True,
        drop_last=False,
        generator=generator,
        **common_loader,
    )
    val_loader = DataLoader(
        PupilDataset(val_samples, cfg.image_size, augment=False),
        shuffle=False,
        **common_loader,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = PupilA1Net(cfg.image_size)
    pretrained = load_pretrained_backbone(model, args.pretrained.resolve() if args.pretrained else None)
    mean_xy = np.mean([s.normalized_xy for s in train_samples], axis=0)
    model.coordinate_head.bias.data.copy_(torch.from_numpy(mean_xy.astype(np.float32)))
    model.to(device)
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}")

    backbone = list(model.backbone_parameters())
    if pretrained and cfg.freeze_backbone_epochs > 0:
        for parameter in backbone:
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone,
                "lr": cfg.learning_rate * cfg.backbone_lr_scale,
                "name": "backbone",
            },
            {"params": model.head_parameters(), "lr": cfg.learning_rate, "name": "head"},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.02
    )
    criterion = nn.SmoothL1Loss(beta=0.025)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    config_record = {
        **asdict(cfg),
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "pretrained": str(args.pretrained.resolve()) if args.pretrained else None,
        "pretrained_loaded": pretrained,
        "device": str(device),
    }
    (output / "config.json").write_text(
        json.dumps(config_record, indent=2), encoding="utf-8"
    )

    history: list[dict[str, float | int]] = []
    best_error = float("inf")
    stale = 0
    best_path = output / "pupil_a1_best.pth"
    history_csv = output / "history.csv"
    for epoch in range(1, cfg.epochs + 1):
        if pretrained and epoch == cfg.freeze_backbone_epochs + 1:
            for parameter in backbone:
                parameter.requires_grad_(True)
            print(f"Epoch {epoch}: unfroze pretrained backbone")
        model.train()
        train_loss = 0.0
        started = time.time()
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = xy_view(model(images))
                loss = criterion(prediction, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(images)
        scheduler.step()

        metrics = evaluate(model, val_loader, device, width, height)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_samples),
            **metrics,
            "head_lr": optimizer.param_groups[1]["lr"],
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} train={row['train_loss']:.5f} "
            f"val={metrics['val_loss']:.5f} mean={metrics['mean_error_px']:.2f}px "
            f"p95={metrics['p95_error_px']:.2f}px time={row['seconds']:.1f}s"
        )
        with history_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)

        if metrics["mean_error_px"] < best_error:
            best_error = metrics["mean_error_px"]
            stale = 0
            torch.save(
                {"model": model.state_dict(), "config": config_record, "metrics": row},
                best_path,
            )
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"Early stopping: no mean-error improvement for {stale} epochs")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    final_metrics = evaluate(model, val_loader, device, width, height)
    onnx_path = output / "pupil_a1_opset11.onnx"
    onnx_details = export_and_verify(model, onnx_path, cfg.image_size, val_loader, device)
    report = {
        "best_epoch": checkpoint["metrics"]["epoch"],
        "validation_samples": len(val_samples),
        "metrics": final_metrics,
        "onnx": onnx_details,
        "cpu_postprocess": [
            "Read pupil_xy_raw[0, 0, 0, 0] and pupil_xy_raw[0, 1, 0, 0].",
            "Reject non-finite values or values outside [-0.25, 1.25].",
            "Clamp each coordinate to [0, 1] and map back to the eye ROI.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Best epoch {report['best_epoch']}: mean={final_metrics['mean_error_px']:.2f}px, "
        f"median={final_metrics['median_error_px']:.2f}px, "
        f"p95={final_metrics['p95_error_px']:.2f}px"
    )
    print(f"Artifacts: {output}")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    data_root = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, default=data_root / "p1-left")
    parser.add_argument("--right", type=Path, default=data_root / "p1-right")
    parser.add_argument("--output", type=Path, default=here / "outputs_a1")
    parser.add_argument("--pretrained", type=Path, default=here / "pupil_gap.pth")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--temporal-block", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else min(4, os.cpu_count() or 1))
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
