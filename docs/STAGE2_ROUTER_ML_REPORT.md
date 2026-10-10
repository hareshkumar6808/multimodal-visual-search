# Stage 2 Router ML Report

Two lightweight scikit-learn pipelines were trained on the same deterministic split. Both use word unigram/bigram TF-IDF with a 60,000-feature cap and balanced classification. The calibrated LinearSVC uses `CalibratedClassifierCV`; raw SVM margins are never presented as probabilities.

| Candidate | Validation accuracy | Validation macro F1 | Train time | Validation ms/example |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.9808 | 0.9696 | 5.00 s | 0.0934 |
| TF-IDF + calibrated LinearSVC | 0.9823 | **0.9717** | 3.41 s | 0.0986 |

The calibrated LinearSVC won on the primary macro-F1 metric with negligible latency difference. The compressed artifact is 7,812,920 bytes. Validation-selected thresholds are text 0.83, code 0.34, and math 0.91. The threshold search prioritizes accepted accuracy while requiring at least 80% validation coverage where feasible.

Held-out test results: accuracy 0.9834, macro precision 0.9761, macro recall 0.9678, macro F1 0.9717, coverage 0.9486, fallback rate 0.0514, and accepted-route accuracy 0.9965.

| Class | Test F1 | Support |
|---|---:|---:|
| code/java | 1.0000 | 385 |
| code/python | 1.0000 | 389 |
| code/sql | 1.0000 | 379 |
| math/algebra | 0.9545 | 411 |
| math/geometry | 0.9098 | 135 |
| math/probability | 0.8839 | 117 |
| text/biomedical | 1.0000 | 148 |
| text/finance | 0.9986 | 365 |
| text/legal | 0.9987 | 376 |

On 1,000 warmed single-example predictions, classifier latency was mean 1.477 ms, p50 1.428 ms, and p95 1.754 ms on the development machine. Runtime additionally reports feature-build, classifier, and total router time per request. The full confusion matrix and exact report are reproducible in ignored `.tools/stage2/reports/router_evaluation.json`.
