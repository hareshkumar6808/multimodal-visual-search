"""Evaluate the frozen production router once on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / ".tools/stage2/data/router/test.jsonl")
    parser.add_argument(
        "--model", type=Path, default=ROOT / "services/orchestrator/artifacts/task_router_v1.joblib"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".tools/stage2/reports/router_evaluation.json"
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()]
    model_data = joblib.load(args.model)
    model = model_data["pipeline"]
    texts, labels = [r["text"] for r in rows], [r["label"] for r in rows]
    probabilities = model.predict_proba(texts)
    classes = np.asarray(model.classes_)
    predictions, accepted = [], []
    for probability, actual in zip(probabilities, labels, strict=True):
        family = actual.split("/", 1)[0]
        columns = [i for i, label in enumerate(classes) if label.startswith(f"{family}/")]
        column = max(columns, key=lambda index: probability[index])
        confidence = float(probability[column])
        is_accepted = confidence >= model_data["thresholds"][family]
        predictions.append(str(classes[column]) if is_accepted else f"{family}/general")
        accepted.append(is_accepted)
    raw_predictions = model.predict(texts)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, raw_predictions, average="macro", zero_division=0
    )
    accepted_indexes = [i for i, value in enumerate(accepted) if value]
    report = {
        "accuracy": accuracy_score(labels, raw_predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "coverage": sum(accepted) / len(accepted),
        "fallback_rate": 1 - sum(accepted) / len(accepted),
        "accepted_route_accuracy": sum(raw_predictions[i] == labels[i] for i in accepted_indexes)
        / max(len(accepted_indexes), 1),
        "per_class": classification_report(
            labels, raw_predictions, output_dict=True, zero_division=0
        ),
        "labels": list(classes),
        "confusion_matrix": confusion_matrix(labels, raw_predictions, labels=classes).tolist(),
        "artifact_size_bytes": args.model.stat().st_size,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"per_class", "confusion_matrix"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
