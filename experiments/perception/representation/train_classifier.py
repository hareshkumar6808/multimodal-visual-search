"""Train and evaluate MobileNetV3-Small on prepared DocLayNet proxy labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts.models import RepresentationKind  # noqa: E402
from services.perception.representation.mobilenet import (  # noqa: E402
    REPRESENTATION_CLASSES,
)

RepresentationClass = RepresentationKind


class ManifestImages(Dataset[tuple[Any, int]]):
    def __init__(self, root: Path, rows: list[dict[str, str]], transform: Any) -> None:
        self.root = root
        self.rows = rows
        self.transform = transform
        self.class_index: dict[RepresentationClass, int] = {
            name: index for index, name in enumerate(REPRESENTATION_CLASSES)
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as source:
            image = source.convert("RGB")
        label = cast(RepresentationClass, row["label"])
        return self.transform(image), self.class_index[label]


def read_manifest(path: Path) -> tuple[Path, dict[str, list[dict[str, str]]]]:
    root = path.parent
    grouped: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        grouped[row["split"]].append(row)
    return root, grouped


def _metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    count = len(REPRESENTATION_CLASSES)
    matrix = [[0 for _ in range(count)] for _ in range(count)]
    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[actual][predicted] += 1
    per_class: dict[str, Any] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    for index, name in enumerate(REPRESENTATION_CLASSES):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    pixel_idx = REPRESENTATION_CLASSES.index("pixels_required")
    pixel_support = sum(matrix[pixel_idx])
    false_pixel_discard = sum(
        matrix[pixel_idx][index]
        for index, name in enumerate(REPRESENTATION_CLASSES)
        if name in {"text_sufficient", "structure_sufficient"}
    )
    total = len(y_true)
    return {
        "sample_count": total,
        "accuracy": sum(matrix[index][index] for index in range(count)) / total if total else 0.0,
        "macro_precision": sum(precisions) / count,
        "macro_recall": sum(recalls) / count,
        "macro_f1": sum(f1_values) / count,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(REPRESENTATION_CLASSES),
            "rows_true_columns_predicted": matrix,
        },
        "false_pixel_discard_rate": (
            false_pixel_discard / pixel_support if pixel_support else None
        ),
        "false_pixel_discard_count": false_pixel_discard,
        "pixel_required_support": pixel_support,
    }


def _evaluate(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> dict[str, Any]:
    model.eval()
    actuals: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            actuals.extend(labels.tolist())
    return _metrics(actuals, predictions)


def train(
    manifest_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    class_weighting: bool = True,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root, grouped = read_manifest(manifest_path)
    train_counts = Counter(row["label"] for row in grouped["train"])
    absent = [name for name in REPRESENTATION_CLASSES if not train_counts[name]]
    if absent:
        raise ValueError(
            f"training split lacks proxy classes: {absent}; counts={dict(train_counts)}"
        )
    if not grouped["validation"] or not grouped["test"]:
        raise ValueError("dataset must provide non-empty validation and held-out test splits")

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_data = ManifestImages(root, grouped["train"], preprocess)
    validation_data = ManifestImages(root, grouped["validation"], preprocess)
    test_data = ManifestImages(root, grouped["test"], preprocess)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator
    )
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(REPRESENTATION_CLASSES))
    model.to(device)
    counts_by_index = [train_counts[name] for name in REPRESENTATION_CLASSES]
    weights_tensor = torch.tensor(
        [
            len(train_data) / (len(REPRESENTATION_CLASSES) * max(1, item))
            for item in counts_by_index
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights_tensor if class_weighting else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epoch_log: list[dict[str, float | int]] = []
    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            sample_count += len(labels)
        validation_metrics = _evaluate(model, validation_loader, device)
        epoch_log.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(1, sample_count),
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )
        if float(validation_metrics["macro_f1"]) >= best_f1:
            best_f1 = float(validation_metrics["macro_f1"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    training_seconds = time.perf_counter() - training_start
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")

    model.load_state_dict(best_state, strict=True)
    model.eval()
    test_metrics = _evaluate(model, test_loader, device) if evaluate_test else None
    checkpoint = {
        "format_version": 1,
        "architecture": "mobilenet_v3_small",
        "classes": list(REPRESENTATION_CLASSES),
        "state_dict": best_state,
        "pretrained_weights": weights.url,
        "best_validation_epoch": best_epoch,
        "validation_macro_f1": best_f1,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    summary_path = manifest_path.parent / "summary.json"
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = {
        "architecture": "MobileNetV3-Small",
        "seed": seed,
        "requested_epochs": epochs,
        "class_weighted_loss": class_weighting,
        "input_size": [224, 224],
        "augmentation": "none",
        "test_evaluation": "completed" if evaluate_test else "not run; validation-only experiment",
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "pretrained_weights_url": weights.url,
        "dataset": dataset_summary,
        "train_counts": {name: train_counts[name] for name in REPRESENTATION_CLASSES},
        "split_counts": {key: len(value) for key, value in grouped.items()},
        "epoch_metrics": epoch_log,
        "held_out_test": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": checkpoint_digest,
        "limitations": [
            "Targets are deterministic proxies derived from DocLayNet layout annotations.",
            "This small document-page dataset does not represent arbitrary desktop screenshots.",
        ],
    }
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/perception/dataset/doclaynet-small/manifest.jsonl"),
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="do not evaluate the held-out test split; use for model ablations",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/perception/mobilenet_v3_small.pth"),
    )
    parser.add_argument(
        "--metrics", type=Path, default=Path("models/perception/evaluation.json")
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--class-weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use inverse-frequency class-weighted cross entropy",
    )
    args = parser.parse_args()
    report = train(
        args.manifest,
        args.checkpoint,
        args.metrics,
        args.epochs,
        args.batch_size,
        args.seed,
        args.class_weighting,
        not args.validation_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
