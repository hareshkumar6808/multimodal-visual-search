"""Train and select one calibrated lightweight semantic router."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / ".tools/stage2/data/router"
MODEL_DIR = ROOT / ".tools/stage2/models/router"
PRODUCTION = ROOT / "services/orchestrator/artifacts/task_router_v1.joblib"


def read(path: Path) -> tuple[list[str], list[str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row["text"] for row in rows], [row["label"] for row in rows]


def thresholds(model: Pipeline, texts: list[str], labels: list[str]) -> dict[str, float]:
    probabilities = model.predict_proba(texts)
    classes = np.asarray(model.classes_)
    output: dict[str, float] = {}
    for family in ("text", "code", "math"):
        indexes = [i for i, label in enumerate(labels) if label.startswith(f"{family}/")]
        family_cols = [i for i, label in enumerate(classes) if label.startswith(f"{family}/")]
        best = (0.0, 0.0, 0.34)
        for threshold in np.arange(0.34, 0.91, 0.01):
            accepted = []
            correct = 0
            for i in indexes:
                col = max(family_cols, key=lambda value: probabilities[i, value])
                if probabilities[i, col] >= threshold:
                    accepted.append(i)
                    correct += int(classes[col] == labels[i])
            coverage = len(accepted) / max(len(indexes), 1)
            accuracy = correct / max(len(accepted), 1)
            score = accuracy if coverage >= 0.80 else accuracy * coverage / 0.80
            if (score, coverage) > best[:2]:
                best = (score, coverage, float(round(threshold, 2)))
        output[family] = best[2]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--production", type=Path, default=PRODUCTION)
    args = parser.parse_args()
    x_train, y_train = read(args.data / "train.jsonl")
    x_val, y_val = read(args.data / "validation.jsonl")
    candidates = {
        "tfidf_logistic_regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", C=4.0
        ),
        "tfidf_calibrated_linear_svc": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", C=1.0), cv=3
        ),
    }
    reports: dict[str, dict[str, float]] = {}
    trained: dict[str, Pipeline] = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, classifier in candidates.items():
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=60000
                    ),
                ),
                ("classifier", classifier),
            ]
        )
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predictions = pipeline.predict(x_val)
        latency_ms = (time.perf_counter() - started) * 1000 / max(len(x_val), 1)
        reports[name] = {
            "accuracy": accuracy_score(y_val, predictions),
            "macro_f1": f1_score(y_val, predictions, average="macro"),
            "train_seconds": train_seconds,
            "validation_ms_per_example": latency_ms,
        }
        trained[name] = pipeline
    winner = max(
        reports,
        key=lambda name: (reports[name]["macro_f1"], -reports[name]["validation_ms_per_example"]),
    )
    artifact = {
        "artifact_version": 1,
        "pipeline": trained[winner],
        "thresholds": thresholds(trained[winner], x_val, y_val),
        "winner": winner,
        "validation": reports[winner],
        "candidates": reports,
    }
    args.production.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.production, compress=3)
    (MODEL_DIR / "training_report.json").write_text(
        json.dumps({**reports, "winner": winner, "thresholds": artifact["thresholds"]}, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"winner": winner, "candidates": reports, "thresholds": artifact["thresholds"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
