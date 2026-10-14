"""Export MobileNet to ONNX and retain only if parity and CPU speed justify it."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.perception.representation.mobilenet import REPRESENTATION_CLASSES  # noqa: E402


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


def export_compare(
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    sample_limit: int,
    repeats: int,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort  # type: ignore[import-untyped]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features, len(REPRESENTATION_CLASSES)
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    torch.onnx.export(
        model,
        (example,),
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(output_path))
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    manifest_root = manifest_path.parent
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["split"] == "test"
    ][:sample_limit]
    tensors: list[np.ndarray[Any, Any]] = []
    for row in rows:
        with Image.open(manifest_root / row["image"]) as source:
            tensor = transform(source.convert("RGB")).unsqueeze(0)
        tensors.append(tensor.numpy())
    if not tensors:
        raise ValueError("no held-out samples available for ONNX comparison")

    for tensor in tensors[: min(5, len(tensors))]:
        with torch.inference_mode():
            model(torch.from_numpy(tensor))
        session.run(None, {"image": tensor})

    torch_ms: list[float] = []
    onnx_ms: list[float] = []
    max_abs_differences: list[float] = []
    argmax_matches: list[bool] = []
    for tensor in tensors:
        with torch.inference_mode():
            torch_logits = model(torch.from_numpy(tensor))
            torch_probabilities = torch.softmax(torch_logits, dim=1).numpy()
        onnx_logits = session.run(None, {"image": tensor})[0]
        onnx_probabilities = torch.softmax(torch.from_numpy(onnx_logits), dim=1).numpy()
        max_abs_differences.append(
            float(np.max(np.abs(torch_probabilities - onnx_probabilities)))
        )
        argmax_matches.append(
            int(torch_probabilities.argmax()) == int(onnx_probabilities.argmax())
        )
        for _ in range(repeats):
            start = time.perf_counter()
            with torch.inference_mode():
                model(torch.from_numpy(tensor))
            torch_ms.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            session.run(None, {"image": tensor})
            onnx_ms.append((time.perf_counter() - start) * 1000)

    torch_median = statistics.median(torch_ms)
    onnx_median = statistics.median(onnx_ms)
    max_abs = max(max_abs_differences)
    all_argmax_match = all(argmax_matches)
    reliable = max_abs <= 1e-4 and all_argmax_match
    speedup = torch_median / onnx_median if onnx_median else 0.0
    useful = speedup >= 1.10
    retained = reliable and useful
    if not retained:
        output_path.unlink(missing_ok=True)

    return {
        "sample_count": len(tensors),
        "repeats_per_sample": repeats,
        "parity": {
            "max_absolute_probability_difference": max_abs,
            "argmax_agreement_rate": sum(argmax_matches) / len(argmax_matches),
            "reliable_threshold": 1e-4,
            "reliable": reliable,
        },
        "latency_ms": {
            "pytorch_cpu_median": torch_median,
            "pytorch_cpu_p95": _percentile(torch_ms, 0.95),
            "onnx_runtime_cpu_median": onnx_median,
            "onnx_runtime_cpu_p95": _percentile(onnx_ms, 0.95),
            "onnx_speedup": speedup,
        },
        "useful_speedup_threshold": 1.10,
        "useful": useful,
        "retained": retained,
        "onnx_path": str(output_path) if retained else None,
        "onnx_bytes": output_path.stat().st_size if retained else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/perception/mobilenet_v3_small.pth")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/perception/dataset/doclaynet-small/manifest.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/perception/cache/mobilenet_v3_small.onnx")
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    report = export_compare(
        args.checkpoint, args.manifest, args.output, args.samples, args.repeats
    )
    report_path = Path("models/perception/onnx_comparison.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
