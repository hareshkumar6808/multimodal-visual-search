# Stage 2 perception latency and deployment

## Fresh measurements

Measurements used ONNX Runtime CPU on this Windows host. The 20-image, 200-repeat selector benchmark measured 135.99 ms cold start, 11.26 ms warm median, and 17.04 ms warm p95. Full selector decision median was 11.24 ms and p95 17.67 ms. Process working set rose by 40,001,536 bytes after cold load and inference; that figure includes Python, Pillow, and ONNX Runtime.

A paired controlled pipeline run used 30 validation images with five repetitions and identical deterministic zero-I/O OCR. The Stage 1-equivalent path measured 16.66 ms median / 24.39 ms p95. Stage 2 measured 28.59 ms / 43.02 ms. Paired added overhead was 11.33 ms median / 19.64 ms p95. The percentage increase looks large only because OCR was intentionally removed to isolate selector cost.

The most recent real-Tesseract benchmark is historical because Tesseract was unavailable for this continuation run: OCR-only median 1,415.15 ms; sequential OCR plus selector median 1,432.61 ms; concurrent median 1,546.28 ms. Concurrency was slower and remains disabled. These runs are machine-specific and should not be presented as fresh measurements.

## ONNX parity

Fresh comparison over 30 pages and 20 repeats per page produced 100% argmax agreement and maximum absolute probability difference `4.5449e-6`. PyTorch median was 6.44 ms and ONNX median 1.07 ms for backend inference, a measured 6.02× speedup. The packaged ONNX file is 6,097,447 bytes, SHA-256 `965b75a574ab348952d001091c1a85d59f1f8ed4f02df2649968b1cc85134ad1`.

## Packaging

The built wheel is 11,431,564 bytes and contains both `.onnx` and `.pth` artifacts. It installed into an isolated target and loaded the ONNX model from that installed tree; four probabilities were returned and summed to 1.0 without network access. The wheel SHA-256 for this build is `5ad26fcfd7d86b48ee504967795dad0a0be1273ced2c1fcd269a6135041bbf78`.

Machine-readable evidence is in [runtime_benchmark_continuation.json](../models/perception/runtime_benchmark_continuation.json), [perception_overhead.json](../models/perception/perception_overhead.json), [onnx_comparison.json](../models/perception/onnx_comparison.json), and [deployment_verification.json](../models/perception/deployment_verification.json).
