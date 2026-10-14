# Stage 2 representation ML report

## Decision

Retain the frozen MobileNetV3-Small checkpoint as **advisory metadata only**. It must not authorize dropping capture pixels. The query-aware orchestrator remains the final authority and currently preserves pixels for every content request.

Fresh reproduction matched the committed DocLayNet-small proxy results exactly. The broader 280-image cross-source benchmark then showed severe domain shift: accuracy 0.2107, macro F1 0.2040, and 142/170 (83.53%) critical proxy examples classified as a discard class. Incorrect predictions were at least as confident as correct ones. This evidence rejects a threshold-only deployment policy and does not justify a larger model or retraining on the same weak labels.

## Baseline and selected checkpoint

The baseline and final selected checkpoint are the same class-weighted, seed-1729 model. No candidate replaced it.

| Evaluation | Accuracy | Macro F1 | Critical false discard |
| --- | ---: | ---: | ---: |
| Validation, raw model, 63 DocLayNet proxy pages | 0.6190 | 0.5251 | 3/9 (33.33%) |
| Historical test diagnostic, 49 pages | 0.8163 | 0.6968 | 2/5 (40.00%) |
| Independent cross-source proxies, 280 images | 0.2107 | 0.2040 | 142/170 (83.53%) |

The historical test split has been inspected and is diagnostic, not untouched final evidence. All four-class labels are conservative proxies for representation sufficiency rather than task-level human judgments.

## Model and runtime contract

The selector predicts `text_sufficient`, `structure_sufficient`, `pixels_required`, or `mixed`. Input is a 224×224 RGB image normalized with ImageNet statistics. ONNX Runtime is preferred; the PyTorch checkpoint is the local fallback. Missing dependencies, missing or corrupt artifacts, preprocessing failures, invalid scores, and selector exceptions all resolve to a pixel-preserving heuristic result. Runtime performs no model or dataset download.

MIR v0.1 remains accepted. MIR v0.2 adds normalized representation scores, confidence, source/model identity, available representations, and local capability telemetry. Because the classifier does not receive the user query, a `text_sufficient` prediction cannot prove that pixels are unnecessary for the requested task.

## Recommendation

Keep the current artifact for research telemetry and future query-conditioned work. Do not train another architecture until labels are grounded in actual questions and answer equivalence across GUI, code, chart, document, and natural-image domains. The next credible model experiment needs group-held-out task data and a safety objective that penalizes false pixel discard.
