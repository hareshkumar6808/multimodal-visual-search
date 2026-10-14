# Stage 2 representation benchmark

## Reproducibility

The pinned DocLayNet-small archive reproduced at 380,921,569 bytes with SHA-256 `85e289b3635b2e17937e201e567d98097173a7b8116daa72dd9e9fb61d738597`. The prepared set contains 688 train, 63 validation, and 49 test pages. Fresh raw validation and historical test outputs exactly matched the committed metrics and confusion matrices.

The independent benchmark uses capped, pinned public sources and commits no source images: ChartQA charts, CIFAR-10 natural images, a ScreenSpot v2 mirror, RVL-CDIP document examples, and deterministic screenshots rendered from CodeSearchNet Python. Source licenses and exact revisions are recorded in `generalization_benchmark.json`. These labels are deliberately described as proxies: dataset identity does not establish minimum sufficient information for every possible question.

## Cross-source results

| Source | n | Accuracy | Critical false discard |
| --- | ---: | ---: | ---: |
| ChartQA | 30 | 0.3333 | 16/30 (53.33%) |
| CIFAR-10 | 30 | 0.0000 | 30/30 (100%) |
| CodeSearchNet render | 30 | 0.0000 | 30/30 (100%) |
| RVL-CDIP demo | 160 | 0.3000 | 40/50 mixed proxies (80%) |
| ScreenSpot v2 mirror | 30 | 0.0333 | 26/30 (86.67%) |
| **Overall** | **280** | **0.2107** | **142/170 (83.53%)** |

Overall confusion matrix, rows true and columns predicted in `text`, `structure`, `pixels`, `mixed` order:

```text
[[34,13, 6,17],
 [ 9, 9, 9,13],
 [42, 4,10, 4],
 [87, 9, 8, 6]]
```

Calibration is poor: 10-bin ECE 0.5182 and multiclass Brier 1.1923. Incorrect predictions had mean confidence 0.7342; critical errors had mean 0.7904 and median 0.8402. At threshold 0.95, abstention still leaves 26/170 critical false discards while reducing coverage to 12.5%. At 0.99 it leaves 2/170 while coverage falls to 1.43%. Confidence therefore cannot safely grant image omission.

## Degradation

Critical false-discard rates were 84.12% after 50% downscaling, 84.71% for JPEG quality 40, 84.12% with mild blur, 97.06% at 70% brightness, and 84.71% after the HiDPI downsample transform. Prediction agreement with originals ranged from 69.64% to 93.57%. Brightness is the clearest measured instability.

Run the evaluation explicitly with the optional `research` and inference extras installed:

```powershell
python experiments/perception/representation/benchmark_generalization.py --output models/perception/generalization_benchmark.json
```

This command may download pinned evaluation data. Application runtime never calls it. Exact per-class metrics, source metadata, confidence bins, threshold sweeps, degradation matrices, and hashed failure identifiers are in [generalization_benchmark.json](../models/perception/generalization_benchmark.json).
