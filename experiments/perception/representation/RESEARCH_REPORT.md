# Stage 2 research and integration-readiness report

## Executive result

The Stage 2 checkpoint and ONNX export remain the selected artifacts, but the evidence does **not** justify automatic pixel reduction or a production-readiness claim. The target label describes task-specific sufficiency while the training proxies only describe page layout. On visual pages, the model can be confidently wrong. The current selector output is advisory: the orchestrator does not read it, and image forwarding is governed by Stage 1 modality/`visual_required` routing. The perception selector also does not receive the user query. Keep this branch separate for integration review; a future caller must not treat `text_sufficient` or `structure_sufficient` as authorization to discard pixels.

The cost objective is lexicographic rather than a fabricated numeric cost ratio: (1) do not discard/withhold image data based on content-only model output; (2) report false pixel-discard and unnecessary-preservation counts separately; (3) only after the safety constraint, compare extra retention and measured latency. The pipeline currently retains source bytes and the app stores the capture; provider image inclusion is a separate routing choice. A query-aware policy is needed before “sufficient” can mean sufficient for the request.

Final fallback policy: model-load failure, invalid/nonfinite probabilities, image decode/preprocessing failure, and inference exceptions now return `kind=pixels_required` with `pixel_dependency_prior=1.0` and heuristic provenance. Successful model output still passes the existing confidence/visual-conflict guard. This protects fallback paths, but it does not make successful high-confidence classifications query-safe; all output remains advisory until a separate request-aware gate is reviewed.

## Baseline reproduction and failure analysis

The baseline state and hypotheses are recorded in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md). Evaluation script:

```powershell
python experiments/perception/representation/evaluate_policy.py --split validation --output models/perception/research_validation.json
python experiments/perception/representation/evaluate_policy.py --split test --output models/perception/research_test_diagnostic.json
python experiments/perception/representation/benchmark_runtime.py --repeats 100 --samples 20 --output models/perception/runtime_benchmark.json
python experiments/perception/representation/benchmark_ocr_model.py
```

The prepared archive, manifest, and cached ImageNet initialization weights are required to reproduce training without network access. If absent, preparation/training scripts explicitly fetch the pinned dataset/pretrained weights. Neither fetch runs during inference. The validation-only loss ablation used `--no-class-weighting --validation-only` and saved only to ignored `artifacts/perception/cache/`.

The committed MobileNetV3-Small checkpoint (seed 1729; 224x224; ImageNet initialization; class-weighted AdamW; best epoch 8) reproduces the saved test confusion matrix and metrics exactly. Historical test labels are now exposed and are diagnostic only. Nine misclassified test pages were inspected. Two false `pixels_required -> text_sufficient` errors were a magazine page with a large product phone and portrait, and an aviation navigation chart with route markings. Their raw top-class scores were 0.9922 and 0.7770. This is evidence that a probability cutoff cannot generally detect visual evidence the model misses. In this particular test run the independent Stage 1 signal set `visual_required=True`, and the final selector returned `mixed`; this rescued these two examples in the composed policy.

The validation call-path evaluation refutes a blanket claim that this guard is sufficient: 2/9 proxy-critical examples still had composed text/structure predictions. Both were visually dominated pages: a corporate board directory with multiple portraits (`mixed`, composed `text_sufficient`, confidence 0.9731) and a full-page aircraft photograph (`pixels_required`, composed `text_sufficient`, confidence 0.9562). Stage 1 also emitted `visual_required=False` for both. Thus the current full route could withhold image bytes on precisely the cases the proxy calls critical. These examples support label/domain/model limitations; they do not identify a single cause. Resize from a square document page to 224x224, high visual/text co-occurrence, proxy-label mismatch, and model calibration remain hypotheses, not isolated causal findings.

| Split / predictor | Accuracy | Macro F1 | False pixel-discard, pixels_required | False pixel-discard, mixed | Combined critical errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test raw model | 40/49 = 0.8163 | 0.6968 | 2/4 = 50% | 0/1 = 0% | 2/5 = 40% |
| Test heuristic | 10/49 = 0.2041 | 0.1316 | 2/4 = 50% | 1/1 = 100% | 3/5 = 60% |
| Test composed selector | 25/49 = 0.5102 | 0.4006 | 0/4 = 0% | 0/1 = 0% | 0/5 = 0% |
| Validation raw model | 39/63 = 0.6190 | 0.5251 | 1/7 = 14.3% | 2/2 = 100% | 3/9 = 33.3% |
| Validation heuristic | 21/63 = 0.3333 | 0.2393 | 4/7 = 57.1% | 1/2 = 50% | 5/9 = 55.6% |
| Validation composed selector | 27/63 = 0.4286 | 0.3765 | 1/7 = 14.3% | 1/2 = 50% | 2/9 = 22.2% |

All labels are weak DocLayNet proxies. For each critical class, false pixel-discard numerator is true-class pages predicted `text_sufficient` or `structure_sufficient`; denominator is support for that true class. Combined critical rate adds those two numerators and divides by `pixels_required + mixed` supports. “Preserved” means the predicted class was `pixels_required` or `mixed`; “unnecessary preserve” is such a prediction among true `text_sufficient` or `structure_sufficient` proxy labels. These are class-policy counts, not counts of bytes actually deleted or forwarded. Exact per-class metrics, confusion matrices, source-category breakdowns, probabilities, OCR signals, and page hashes are in the two JSON reports. The prior test support is especially weak (4 pixels-required, 1 mixed); zero observed composed errors among five is not evidence of zero real-world risk.

## Data, splits, and candidate-source decision

The local preparation audit found 800 labeled pages from source split totals 691/64/49. Per-class counts are train 224/350/102/12, validation 30/24/7/2, and test 14/30/4/1 for text/structure/pixels/mixed. The source contains six document categories (financial reports, manuals, government tenders, laws/regulations, scientific articles, patents). Current proxy rules map Picture coverage >=20% to `pixels_required`, Picture plus text/structure to `mixed`, Table/Formula/List-item without Picture to `structure_sufficient`, and text-like annotations without either to `text_sufficient`. These rules do not establish task sufficiency; notably any small Picture plus text may become mixed, while a large picture can be treated as pixels-required without knowing the query.

Document-category counts (train/validation/test) are financial_reports 208/18/20, government_tenders 39/1/2, laws_and_regulations 114/12/8, manuals 139/17/7, patents 59/7/2, and scientific_articles 129/8/10. Source-category test support is uneven; category-level raw, heuristic, and composed metrics are broken out in the evaluation JSON, but should not be ranked as robust domain estimates.

Leakage audit: 511 distinct source `original_filename` document groups, no filename shared across official splits, and no shared page hashes. The source splits are group-disjoint by original document in this subset. Collection categories overlap, so this does not establish generalization to new domains. Near-duplicate perceptual-image screening was not performed. No raw source dataset is tracked in Git.

Public dataset review and licensing rationale are in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md). ChartQA (chart reasoning), Screen2Words/RICO (mobile UI), AI2D (diagrams), and DocVQA (document QA) provide useful task-specific or layout signals but no valid automatic map to the four sufficiency classes. Some sources have unclear or restrictive image rights. No new dataset was downloaded and no weak annotation was promoted to ground truth. Data gaps remain for desktop apps, code/terminal screens, web interfaces, charts, natural images, low-resolution/compressed captures, and query-conditioned ambiguity.

## Controlled experiment log

| ID | Hypothesis / change | Data and split | Result | Decision |
| --- | --- | --- | --- | --- |
| BASE-01 | Reproduce the selected baseline before changing it | Existing DocLayNet-small proxy train/validation/test; seed 1729 | Reproduced historical 49-page test metrics and matrix exactly | Keep reports as baseline; historical test is exposed and diagnostic |
| PILOT-01 | Verify the initial transfer-learning pipeline end-to-end | Existing official split; one epoch, CPU; test split | 26.71 s training, test accuracy 0.653, macro F1 0.424, false pixel-discard 3/4 | Historical pipeline smoke run only; not a candidate and further evidence of test split exposure |
| FAIL-01 | Inspect critical false-discard examples and determine if confidence catches them | 9 historical test classifier errors, image inspection | Both `pixels_required` false-text predictions were visually salient; scores 0.9922 and 0.7770 | Confidence alone cannot be treated as proof; do not lower cutoff |
| POLICY-01 | Compare raw, heuristic, and actual composed policy on identical data | Real Tesseract + local ONNX; validation 63 and historical test 49 | Validation composed critical 2/9 vs raw 3/9 and heuristic 5/9. Test composed 0/5 vs raw 2/5 | Heuristic conflict helps but misses two high-confidence validation visual pages; no safe deletion claim |
| CAL-01 | Determine whether max-softmax supports a conservative abstention threshold | Validation only; ECE/Brier and threshold sweep | 10-bin ECE 0.2568, multiclass Brier 0.6275 (n=63). At 0.99 threshold: 57/63 abstentions, 0/9 observed critical errors, 48/54 unnecessary proxy preserves | Do not adopt 0.99: extreme coverage cost and tiny proxy support; high confidence remains uncalibrated |
| LOSS-01 | Test whether inverse-frequency class weighting is necessary | Same seed/config/data, only class weighting disabled; validation selection only; no test evaluation | Unweighted took 235.996 s; raw validation accuracy/F1 0.6667/0.5488 vs weighted 0.6190/0.5251. Both had raw critical errors 3/9 and composed critical errors 2/9. Unweighted composed macro F1 was 0.3714 vs weighted 0.3765; preserved 28/63 vs 29/63 | Reject unweighted checkpoint. Small raw macro-F1 gain did not improve composed safety or composed F1; one seed and proxy minority supports cannot justify replacing baseline |

No larger architecture, oversampling, augmentations, resolution change, or extra data was tried. This is intentional: the observed failure points first to invalid/weak targets and absent task conditioning, which a larger network would not resolve. No validation has been treated as a credible final generalization estimate.

## Calibration, safety tradeoff, and robustness

On validation, top-confidence ECE (10 bins) was 0.2568; multiclass Brier score was 0.6275 (63 samples). These estimates are themselves noisy. For raw confidence abstention defined as replacing predictions below threshold with `pixels_required`, the validation sweep is:

| Threshold | Abstain | Critical errors / 9 | Preserved / 63 | Unnecessary preserves / 54 |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0/63 | 3/9 | 7/63 | 1/54 |
| 0.50 | 2/63 | 3/9 | 8/63 | 2/54 |
| 0.65 | 10/63 | 3/9 | 14/63 | 8/54 |
| 0.75 | 20/63 | 3/9 | 22/63 | 16/54 |
| 0.85 | 28/63 | 3/9 | 29/63 | 23/54 |
| 0.90 | 33/63 | 3/9 | 33/63 | 27/54 |
| 0.95 | 41/63 | 3/9 | 41/63 | 35/54 |
| 0.99 | 57/63 | 0/9 | 57/63 | 48/54 |

This threshold sweep is descriptive and does not establish a safe operating point. Confidence thresholding misses the high-confidence visual examples until it preserves almost the entire set. OCR disagreement and actual domain OOD performance remain unmeasured. A deterministic OOD detector has not been validated; unknown-domain inputs must be treated as uncertain by any downstream consumer.

For zero observed critical errors, exact one-sided 95% binomial upper bounds are still large: 0/5 on the composed historical test corresponds to an upper bound of about 45.1%; 0/9 at the validation 0.99 abstention point corresponds to about 28.3%. Zero observed failures does not establish zero real-world risk.

Validation correct/incorrect max-confidence distributions were mean 0.829/0.831 and median 0.858/0.901 (correct n=39, incorrect n=24); an incorrect case reached 0.9920. Stage 1 set `visual_required=False` on 42/63 validation pages, including 5/9 `pixels_required` or `mixed` proxy pages. Actual provider forwarding still depends on the query-specific router; these counts are the independent Stage 1 signal, not a measured provider-upload count.

Deterministic model-only degradation check on all 63 validation pages: native-resolution JPEG quality 40 retained accuracy 39/63=0.619 and macro F1 0.511, with 61/63 prediction agreement; quarter-resolution downsample/upscale plus JPEG quality 40 gave accuracy 38/63=0.603, macro F1 0.477, agreement 53/63, and pixels-required false-discard 2/7 (28.6%). Labels were held fixed proxies, OCR was not rerun, and these are not real screenshot robustness results.

## Runtime and deployment evidence

Existing ONNX comparison: 20 held-out pages, 30 repeats each, PyTorch/ONNX argmax agreement 20/20 and max probability difference `6.03e-6`; median 30.276 ms PyTorch versus 2.861 ms ONNX Runtime CPU (10.58x), p95 49.930 ms versus 4.087 ms. The prior OCR concurrency benchmark measured sequential median 1,432.61 ms vs parallel 1,546.28 ms on 20 images/3 repeats; sequential stays in use. The policy evaluator's full local pipeline timing on validation was median 1,506.3 ms and p95 2,437.4 ms over 63 pages; on historical test, median 1,495.7 ms and p95 2,376.0 ms over 49 pages. These timings include local Tesseract and are machine-specific.

The inference artifact remains 6,097,447-byte ONNX with 6,212,245-byte PyTorch checkpoint fallback. ONNX Runtime CPU is the default when installed; PyTorch and ONNX dependencies are optional extras, and missing optional dependencies fall back to the heuristic. Current separate validation-image benchmark (`benchmark_runtime.py`, 20 images, 100 warm repeats) measured cold first inference at 141.77 ms, warm selector median 14.27 ms and p95 22.45 ms. Windows process working set rose from 50,044,928 to 94,404,608 bytes after load/first inference (peak 100,618,240); this includes Python, Pillow, ONNX Runtime, and the model, not an isolated model allocation. The OCR-included policy evaluator measured median 1,506 ms / p95 2,437 ms on validation (n=63) and 1,496 ms / 2,376 ms on historical test (n=49). Measurements differ by process state and host load; use the run reports for scopes and raw values. Wheel packaging remains unverified because `hatchling` is unavailable in the Python environment. Do not claim deployment packaging complete until a wheel is built and inspected.

## Task-level behavior examples (safety expectations, not measured semantic accuracy)

| Request/content case | Safe expected behavior | Current evidence / limitation |
| --- | --- | --- |
| Question asks for one sentence in a text-only page | OCR may be enough only after query-specific verification; keep source capture available | The selector never sees the query and no answer-equivalence evaluator exists |
| Ask for values or trends in a chart | Preserve pixels and route image to a vision-capable path | Existing router relies on Stage 1 modality and `visual_required`; model metadata is not consumed |
| Ask which table value belongs to which row/column | Preserve image or validate structured extraction before reducing | No table-relation semantic test is available |
| Ask about code indentation or a selected token | Preserve image and OCR/layout together until syntax extraction is checked | OCR text alone can flatten spatial cues; no answer-based test |
| Ask who/what appears in a photograph | Preserve pixels | Visual scene recognition cannot be reconstructed from OCR |
| Mixed text page with portraits or embedded diagram | Preserve pixels for ambiguous/general questions | Board-directory validation error proves Stage 1/model can miss visual content |
| Clear OCR-only extraction request | OCR can be offered as an additional representation, while image remains recoverable | No downstream answer comparison in this repository |
| Ambiguous question or unknown screen type | Preserve pixels / safe abstention | The classifier has no calibrated OOD state and is not query-conditioned |

These are explicit integration scenarios, not an end-to-end semantic benchmark. The project has no independent oracle for whether OCR/layout-only answers preserve task-relevant information.

## Compatibility, tests, and integration readiness

The Stage 2 selector remains behind the existing interface. MIR v0.1 validates with no representation field; MIR v0.2 serializes the four classes, normalized scores, confidence, pixel prior, source/model identity, available representations, and local capabilities. Tests cover score normalization, invalid/nonfinite outputs, visual conflicts, missing/corrupt model fallback, local-only model loading, ONNX parity, and existing compatibility. (Final validation commands and results will be recorded after implementation changes settle.)

Runtime inference performs no dataset or weight download. Training/preparation/downloads remain explicit scripts outside runtime. Privacy: this report and committed artifacts contain model/evaluation data only; raw page images stay in ignored `artifacts/` storage.

### Integration blockers and open work

1. Do not let `text_sufficient`/`structure_sufficient` authorize discarding or not forwarding original pixels. The selector is not query-aware and has two high-confidence visual misses on validation.
2. Query does not reach the perception selector. Add a separately reviewed query-conditioned downstream gate if the product needs to omit visual data; keep content classification and routing separate.
3. Expand evaluation only with verified image rights and labels grounded in task-level QA or reviewed judgments. Need new independent sources and group-held-out evaluation, especially GUI/code/chart/photo/mixed content.
4. Need enough critical-category examples to estimate failure rates with uncertainty intervals; current supports 7 pixels-required + 2 mixed in validation and 4 + 1 in historical test are inadequate.
5. Add calibrated/OOD-aware abstention only when its validation evidence is meaningful; present confidence does not detect the observed visual misses.
6. Record isolated cold-start/peak memory and build/inspect an actual wheel. Optional dependencies, model packaging, and supported CPU/OS portability need installation-level verification.
7. End-to-end answer preservation, image forwarding, and privacy/provider behavior need a coordinated integration test with the routing owner; this work does not implement the other team's router/provider selection.

Current status is research prototype, not production-ready. Baseline is retained unless a controlled validation comparison provides a defensible model improvement. No merge is performed.
