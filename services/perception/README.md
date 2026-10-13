# Context + Multimodal Perception Subsystem (`services/perception`)

The **Multimodal Perception** layer validates screenshot bytes, runs local OCR, profiles content modalities, extracts basic layout elements, scans for privacy flags, and emits MIR v0.2. The contract continues to accept MIR v0.1 payloads.

## Stage 2: Minimum-Sufficient Representation

`services/perception/representation/` defines an injectable `RepresentationSelector` interface and a deterministic `HeuristicRepresentationSelector` fallback. It emits one of `text_sufficient`, `structure_sufficient`, `pixels_required`, or `mixed`, with per-class scores, confidence, pixel-dependency prior, provenance, available representations, and local capabilities. Low confidence, weak OCR, visual content, or mixed evidence conservatively retains pixels. The result is metadata only; this stage does not delete or upload image pixels.

The default implementation is heuristic, not learned. It uses the existing OCR confidence and modality/layout signals. A future model selector can be injected; exceptions fall back to the heuristic. No training or dataset scripts run in the request pipeline. This checkout has no PyTorch/ONNX dependency or verified four-class labeled screenshot dataset, so no classifier was trained and no ML accuracy or latency is claimed. Hardware inventory was unavailable in this environment (Windows management queries denied).

---

## Technical Honesty & Scope Declarations

> [!IMPORTANT]
> - **Heuristic Profiling (Not Machine Learning)**: Modality profiling uses deterministic, transparent heuristic rules based on OCR token keywords, syntax patterns, indentation, and layout bounding boxes. **This is NOT a trained ML classifier.**
> - **Basic Layout Extraction (Not Advanced Semantic Decomposition)**: Object extraction groups OCR bounding boxes into lines, approximate paragraphs, code blocks, table regions, and visual areas based on spatial geometry. **This is NOT learned semantic segmentation, object detection, or vision-language scene reasoning.**
> - **Lightweight Privacy Scanning (Not a Privacy AI)**: Privacy detection uses conservative regex patterns for emails, phone numbers, and obvious API keys/credentials.
> - **Stage 1 Boolean Heuristic (Not Full Visual Dependency Score)**: `visual_required` is a simple deterministic decision rule. **This is NOT the full future Visual Dependency Score, Minimum Sufficient Modality, or Loss-Aware Modality Compression.**
> - **100% Local Processing**: No external cloud APIs (Gemini, NVIDIA, OpenAI) are contacted.

---

## Pipeline Architecture

```
Screenshot Bytes (PNG/JPEG/WebP/BMP) + Context (active_app, window_title, bounds)
                           │
                           ▼
                 1. Local Pre-flight
         (Format magic bytes, dimensions, SHA-256,
          brightness, contrast, sharpness metrics)
                           │
                           ▼
                    2. Real Local OCR
         (Tesseract OCR via pytesseract;
          Windows.Media.Ocr optional verification)
                           │
                           ▼
            3. Deterministic Heuristic Profiling
         (Multi-label scores in [0.0, 1.0] for:
          text, code, table, chart, image, mixed)
                           │
                           ▼
               4. Basic Object Extraction
         (OCR/layout-based grouping: text_block,
          code_block, table_region, visual_region)
                           │
                           ▼
             5. Lightweight Privacy Scanning
         (Email, phone, API key, credential flags)
                           │
                           ▼
           6. Visual Dependency Heuristic
         (visual_required: True if charts/images/low-OCR;
          False if clean structured text/code/table)
                           │
                           ▼
              7. MIR v0.2 with optional representation metadata
```

---

## MIR Schema Compatibility

The subsystem outputs JSON-serializable MIR v0.2. Existing MIR v0.1 payloads remain valid; `representation` is optional so older callers and fixtures continue to validate. The additional field has this shape:

```json
{
  "mir_version": "0.2",
  "request_id": "req-12345",
  "image": {
    "sha256": "3a7b9c...",
    "width": 800,
    "height": 500
  },
  "context": {
    "active_app": "Visual Studio Code",
    "window_title": "main.py",
    "selection_mode": "rectangle",
    "bounds": {
      "x": 100,
      "y": 100,
      "width": 800,
      "height": 500
    }
  },
  "modalities": {
    "text": 0.40,
    "code": 0.95,
    "table": 0.00,
    "chart": 0.00,
    "image": 0.10,
    "mixed": 0.00
  },
  "primary_modality": "code",
  "ocr": {
    "text": "def calculate_total(items):\n    return sum(items)",
    "confidence": 0.96,
    "regions": [
      {
        "bbox": { "x": 12, "y": 15, "width": 80, "height": 18 },
        "text": "def",
        "confidence": 0.98
      }
    ]
  },
  "objects": [
    {
      "id": "obj_1",
      "type": "code_block",
      "bbox": { "x": 12, "y": 15, "width": 400, "height": 60 },
      "confidence": 0.96,
      "content": "def calculate_total(items):\n    return sum(items)",
      "metadata": { "line_count": 2 }
    }
  ],
  "visual_required": false,
  "privacy_flags": [],
  "overall_confidence": 0.94,
  "representation": {
    "kind": "structure_sufficient",
    "scores": {
      "text_sufficient": 0.86,
      "structure_sufficient": 0.90,
      "pixels_required": 0.10,
      "mixed": 0.00
    },
    "confidence": 0.94,
    "pixel_dependency_prior": 0.10,
    "source": "heuristic",
    "model": null,
    "available_representations": ["pixels", "ocr_text", "layout"],
    "local_capabilities": {"ocr": true, "layout": true, "ml_model": false}
  }
}
```

*Note on image object:* Internal preflight computes `aspect_ratio`, `brightness`, `contrast`, and `sharpness`. However, to strictly maintain compatibility with the canonical MIR v0.1 contract, `aspect_ratio` is **not** included in the canonical `image` object (`sha256`, `width`, `height`).

---

## Local OCR Configuration

* **Engine**: Tesseract OCR (v5.5.3 or compatible) accessed via `pytesseract`.
* **Standard Windows Executable Path**:
  ```text
  C:\Program Files\Tesseract-OCR\tesseract.exe
  ```
* **Auto-Discovery**: The engine checks system `PATH` first. If `tesseract` is not in `PATH`, it automatically defaults to `C:\Program Files\Tesseract-OCR\tesseract.exe`.
* **Custom Path Override**: Can be provided directly when calling `analyze_capture(..., tesseract_path=r"...")` or when instantiating `TesseractOCREngine(executable_path=r"...")`.
* **Windows.Media.Ocr Note**: An optional engine is provided in `services/perception/ocr/windows_media_engine.py`. It checks at runtime whether Python WinRT bindings (`winsdk` or `winrt`) are installed. Because standard Windows Python 3.12 installs lack WinRT bindings, it reports `is_available() -> False` and safely defers to Tesseract as the verified working local engine.

---

## Inter-Service Integration

Another service (such as `services/orchestrator/`) can invoke the perception subsystem cleanly in one line:

```python
from services.perception import analyze_capture

# Synchronous call
mir_dict = analyze_capture(image_bytes, context={"active_app": "Code"})

# Returns a standard, fully JSON-serializable dictionary matching MIR v0.2
```

Or instantiate a standalone pipeline:

```python
from services.perception import PerceptionPipeline

pipeline = PerceptionPipeline()
mir_model = pipeline.analyze(image_bytes, context=context, request_id="custom-id")

# Inspect pipeline execution telemetry:
print(pipeline.last_telemetry)
# {
#   "preflight_ms": 1,
#   "ocr_ms": 42,
#   "profiling_ms": 2,
#   "extraction_ms": 1,
#   "privacy_ms": 0,
#   "total_pipeline_ms": 46
# }
```

---

## Heuristic Profiling Rules Catalog

1. **Text**:
   - Word count thresholds ($\ge 50$ words $\rightarrow 0.85$, $\ge 20$ words $\rightarrow 0.65$).
   - Text density / coverage relative to screenshot dimensions.
2. **Code**:
   - Keywords across Python (`def`, `class`, `import`, `self`, `return`), Java/C/C++ (`public`, `void`, `#include`, `std::`, `nullptr`), JavaScript/TypeScript (`const`, `function`, `=>`, `interface`, `===`), and SQL (`SELECT`, `FROM`, `WHERE`, `JOIN`).
   - Structural code symbols (`{`, `}`, `;`, `->`, `//`, `!=`, indentation $\ge 4$ spaces).
   - Window title / active app hints (`Code`, `.py`, `.ts`, `Visual Studio`).
3. **Table**:
   - Delimiters (Markdown pipes `|`, ASCII grids `+---+`, tabs `\t`).
   - Spatial vertical alignment: repeated X coordinates across multiple Y baselines.
   - Application hints (`excel`, `sheets`, `calc`, `grid`).
4. **Chart**:
   - Terminology (`chart`, `graph`, `plot`, `axis`, `legend`, `scatter`, `trend`, `%`).
   - Geometric layout of tick numbers / percentage signs around graphical regions.
5. **Image**:
   - Low OCR text coverage ($< 0.05$).
   - High visual area without character detections.
6. **Mixed**:
   - Triggered when two or more distinct modalities have significant concurrent representation (top score $\ge 0.40$ and second score $\ge 0.35$).

---

## Known Weaknesses & Future Improvements

1. **OCR Dependent on Font Size & DPI**: Extremely small or stylized typography on high-DPI displays may yield low OCR confidence. Future work should add multi-scale image pyramid OCR pre-processing.
2. **Heuristic Collision**: Code comments containing extensive natural language prose may slightly elevate text scores.
3. **Future Learned Classifier (Phase 2/3)**: Stage 1 deliberately avoids machine learning to ensure 100% deterministic local predictability. In Phase 2/3, a lightweight quantized local vision model (e.g. MobileNet / FastViT) will be evaluated against this deterministic baseline.
4. **Visual Dependency Score**: Phase 2 will introduce an information-loss estimation metric comparing tokenized text entropy with image pixel entropy.

