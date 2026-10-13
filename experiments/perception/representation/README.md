# Learned representation experiment

## Data and labels

The reproducible preparation script pins `pierreguillou/DocLayNet-small` revision `389e375`. The source card identifies DocLayNet v1.1.0, upstream CDLA-Permissive-1.0 terms, and official train/validation/test splits. The small repository's own dataset metadata says `other`; consult the upstream license before redistribution. The downloaded archive was 380,921,569 bytes (SHA-256 `85e289b3635b2e17937e201e567d98097173a7b8116daa72dd9e9fb61d738597`). See the [dataset card](https://huggingface.co/datasets/pierreguillou/DocLayNet) and [upstream license](https://github.com/DS4SD/DocLayNet/blob/main/LICENSE).

The preparation uses existing page images and `form` annotations. It maps Picture coverage >=20% to `pixels_required`, Picture plus other annotated content to `mixed`, Table/Formula/List-item without Picture to `structure_sufficient`, and remaining text-like annotated pages to `text_sufficient`. Four of 804 source pages lacked usable annotations and were skipped. These are proxy labels, not human judgments of minimum sufficient information. Dataset images and cache are ignored under `artifacts/perception/` and are not included in the application package.

| Split | Total | text_sufficient | structure_sufficient | pixels_required | mixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 688 | 224 | 350 | 102 | 12 |
| Validation | 63 | 30 | 24 | 7 | 2 |
| Test | 49 | 14 | 30 | 4 | 1 |

## Reproduction

Install the optional `perception-ml` extra, then run:

```powershell
python experiments/perception/representation/prepare_dataset.py
python experiments/perception/representation/train_classifier.py --epochs 8 --batch-size 16
python experiments/perception/representation/export_onnx.py
python experiments/perception/representation/benchmark_ocr_model.py
```

Data download and pretrained ImageNet weight download are explicit experiment-time operations. Runtime code loads only packaged local artifacts and never fetches data or weights. Training uses seed 1729, the official fixed split, pretrained torchvision MobileNetV3-Small, class-weighted cross entropy, and validation macro-F1 checkpoint selection. It automatically uses CUDA if available; the recorded run used CPU.

## Recorded training and evaluation

Recorded environment: Python 3.12.10, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, Windows 11, CPU-only, 16 logical CPUs, reported physical memory 16,909,131,776 bytes. No CUDA device was available. Training used an 8-thread cap and took **384.17 seconds** for 8 epochs. Best checkpoint was epoch 8.

Held-out test (49 examples): accuracy **0.8163**, macro precision **0.6577**, macro recall **0.7786**, macro F1 **0.6968**. Per-class F1: text_sufficient **0.6897**, structure_sufficient **0.9310**, pixels_required **0.5000**, mixed **0.6667**. Confusion matrix rows are actual classes and columns predicted, in the class order above:

```text
[[10, 1, 2, 1],
 [ 3,27, 0, 0],
 [ 2, 0, 2, 0],
 [ 0, 0, 0, 1]]
```

False pixel-discard rate was **2/4 = 50%**: two true `pixels_required` test pages were predicted as `text_sufficient` or `structure_sufficient`. The `mixed` class retains pixels. Four positive pixel-required samples and one mixed sample are far too few to establish a safe discard policy. The runtime therefore applies confidence and visual-evidence gates and falls back conservatively; this does not erase the measured error or prove safety on new data. Document-page bias, proxy-label noise, small evaluation size, and domain shift to desktop screenshots are substantial limitations.

## Artifacts and runtime comparison

The tracked artifacts are `models/perception/mobilenet_v3_small.pth` (6,212,245 bytes) and `models/perception/mobilenet_v3_small.onnx` (6,097,447 bytes). `evaluation.json`, `onnx_comparison.json`, and `ocr_model_benchmark.json` contain run details and measurements.

For 20 held-out images over 30 repeats, ONNX Runtime CPU matched PyTorch CPU argmax on all samples (maximum absolute probability difference **6.03e-6**). Median inference was **2.861 ms** ONNX versus **30.276 ms** PyTorch (10.58x measured speedup); p95 was **4.087 ms** versus **49.930 ms**. ONNX was retained because output agreement passed and measured speedup exceeded the export script's 1.10x threshold. Timings depend on this CPU/runtime and do not include the same work as end-to-end OCR.

On 20 held-out images with three paired repetitions using local Tesseract and ONNX, sequential OCR then model had median **1,432.61 ms** (p95 **2,473.91 ms**); concurrent OCR/model had median **1,546.28 ms** (p95 **2,544.88 ms**), a sequential/parallel ratio of **0.926**. Parallel execution was slower in this run and is not used. OCR dominated: OCR-only median was 1,415.15 ms and model selector median was 19.82 ms.

These are small, single-machine measurements, not production benchmarks. Re-run the scripts to assess another machine or runtime. No quantization or broad hyperparameter search was performed.

## Research audit and integration limits

The controlled comparison, validation-only class-weighting ablation, failure inspection, confidence-threshold sweep, degradation check, dataset-source review, and routing limitations are recorded in [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md), with hypotheses and source/license notes in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) and a structured log in [`experiment_log.json`](experiment_log.json). The Stage 2 selector never receives the request query and the orchestrator does not consume its representation output; keep predictions advisory and do not use them as permission to omit original pixels. The historical test split is now exposed and must not be presented as an untouched final holdout.

Reproduce policy/calibration and systems evidence with:

```powershell
python experiments/perception/representation/evaluate_policy.py --split validation --output models/perception/research_validation.json
python experiments/perception/representation/evaluate_policy.py --split test --output models/perception/research_test_diagnostic.json
python experiments/perception/representation/benchmark_runtime.py --repeats 100 --samples 20 --output models/perception/runtime_benchmark.json
```

The test command is for a historical diagnostic only. Validation data supports exploratory comparisons but remains too small and proxy-labeled to establish deployment safety. The original class-weighted checkpoint is retained: a single-seed unweighted loss ablation did not reduce critical errors under the composed selector.
