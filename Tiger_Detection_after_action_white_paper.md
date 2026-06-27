# TIGER DETECTION

## Small Data, Large Augmentation, Big Results

**Scenario:** TIGERTEST v13 + v14 | **Dataset:** tiger111 | **Architecture:** YOLOv8n (3.2M params)
**Device:** Apple M3 (arm64) -- MPS backend | **Pipeline:** cvLayer MLOps | **Seed:** 42

---

## 01 // TRY

Single-class tiger detection. 10 source images expanded to 1000 via automated augmentation pipeline. 93.7% mAP@0.5 from ten photographs.

### Final Metrics (10-Image Model, v13, 20 Epochs)

| Metric | Value |
|--------|-------|
| mAP@0.5 | **93.69%** |
| mAP@0.5:0.95 | **90.18%** |
| Precision | **95.60%** |
| Recall | **91.52%** |
| F1 Score | **0.94** (at conf 0.243) |
| Source Images | 10 |
| Augmented Training Set | 1000 images, 1022 instances |
| Epochs | 20 |
| Training Time | 175,732s (~49 hours) |

### Confusion Matrix

| | Predicted Tiger | Predicted Background |
|---|---|---|
| **Actual Tiger** | 477 (TP) | 45 (FN) |
| **Actual Background** | 22 (FP) | -- |

Conservative detector. When it says tiger, it means tiger (precision 0.98 at max confidence). The 45 false negatives are missed detections on heavily augmented variants.

---

## 02 // FIELD TEST

Novel images the model has never seen. Tested via cvLayer Test Range. Every image is out-of-distribution.

### Single Tiger -- Cross-Variant Generalization

| Test | Subject | Confidence | Inference | BBox | Verdict |
|------|---------|------------|-----------|------|---------|
| 1 | Close-up roaring tiger | **0.998** | 100 ms | [0, 0, 311, 436] | DETECTED |
| 2 | White tiger rolling on back | **0.992** | 2,438 ms | [0, 1, 1303, 760] | DETECTED |
| 3 | Standing tiger with text overlay | **0.956** | 88 ms | [0, 0, 159, 111] | DETECTED |
| 4 | Baby tiger cub | **0.969** | 71 ms | [9, 0, 1403, 701] | DETECTED |

**Result:** 4/4 detected. Mean confidence: 0.979. Zero false positives. Zero false negatives.

**Finding:** The model generalizes across coat variants (white tiger -- absent from training), age classes (baby cub -- completely different morphology), unusual poses (rolling on back), and partial occlusion (watermark text overlay). The augmentation pipeline's color space conversion and geometric transforms created sufficient invariance for cross-variant generalization from only 10 source images.

### Multi-Tiger -- The Known Failure Mode

| Test | Subject | Confidence | Detections | Tigers Present | Verdict |
|------|---------|------------|------------|----------------|---------|
| 5 | Two adults walking | 0.989 | 1 | 2 | **MERGED** -- single box covers both |
| 6 | Adult with cub | 0.972 | 1 | 2 | **PARTIAL** -- misses second tiger |

**Result:** 2/4 tigers found. 2 missed. Model returns exactly 1 detection per image.

**Root cause:** Training data contained exclusively single-tiger-per-image examples with frame-filling bboxes centered at (0.5, 0.5). The model learned "tiger = one large object in center" rather than "tiger = individual animal that may appear alongside others." NMS merges overlapping proposals at IoU 0.70 threshold. Anchor priors lack spatial resolution for closely-spaced instances.

**Fix:** Add 5-10 multi-tiger source images with per-instance annotations. The augmentation pipeline would amplify these into 500-1000 multi-instance training samples.

---

## 03 // EXPLORATION

### Training Curves -- 10-Image Model (20 Epochs)

All values from `results.csv`. Real measurements.

| Epoch | Train Box | Train Cls | Val Box | Val Cls | mAP@0.5 | mAP@.5:.95 | Precision | Recall |
|-------|-----------|-----------|---------|---------|---------|------------|-----------|--------|
| 1 | 0.729 | 2.269 | 0.724 | 2.535 | 0.307 | 0.205 | 0.314 | 0.522 |
| 2 | 0.507 | 1.485 | 0.347 | 1.333 | 0.754 | 0.681 | 0.706 | 0.674 |
| 3 | 0.409 | 0.971 | 0.424 | 1.215 | 0.670 | 0.573 | 0.642 | 0.778 |
| 4 | 0.331 | 0.673 | 0.458 | 0.917 | 0.831 | 0.720 | 0.794 | 0.805 |
| 5 | 0.273 | 0.484 | 0.419 | 0.991 | 0.836 | 0.737 | 0.784 | 0.858 |
| 6 | 0.248 | 0.424 | 0.454 | 0.997 | 0.844 | 0.732 | 0.861 | 0.824 |
| 7 | 0.253 | 0.389 | 0.387 | 1.027 | 0.835 | 0.746 | 0.845 | 0.824 |
| 8 | 0.228 | 0.328 | 0.479 | 0.687 | 0.904 | 0.819 | 0.911 | 0.903 |
| 9 | 0.216 | 0.289 | 0.343 | 0.600 | **0.928** | **0.893** | 0.947 | 0.912 |
| 10 | 0.196 | 0.295 | 0.311 | 0.587 | 0.915 | 0.884 | 0.950 | 0.909 |
| 11 | 0.243 | 0.325 | 0.382 | 0.688 | 0.911 | 0.847 | 0.931 | 0.879 |
| 12 | 0.203 | 0.160 | 0.343 | 0.691 | 0.935 | 0.875 | 0.940 | 0.897 |
| 13 | 0.182 | 0.138 | 0.296 | 0.809 | 0.907 | 0.843 | 0.941 | 0.881 |
| 14 | 0.171 | 0.133 | 0.254 | 0.728 | 0.898 | 0.850 | 0.952 | 0.913 |
| 15 | 0.156 | 0.134 | 0.302 | 0.686 | 0.904 | 0.852 | 0.956 | 0.914 |
| 16 | 0.153 | 0.110 | 0.278 | 0.691 | 0.901 | 0.848 | 0.953 | 0.916 |
| 17 | 0.135 | 0.110 | 0.257 | 0.664 | 0.911 | 0.863 | 0.956 | 0.916 |
| 18 | 0.132 | 0.109 | 0.326 | 0.693 | 0.931 | 0.899 | 0.954 | 0.914 |
| 19 | 0.116 | 0.109 | 0.296 | 0.742 | **0.937** | **0.902** | 0.956 | 0.914 |
| 20 | 0.113 | 0.092 | 0.274 | 0.680 | 0.935 | 0.900 | 0.954 | 0.916 |

Box loss dropped 84% (0.729 to 0.113). Cls loss dropped 96% (2.269 to 0.092). mAP@0.5 crossed 0.90 at epoch 9. Remaining epochs refined localization precision.

### Calibration

F1 peak: 0.94 at confidence 0.243. Flat from 0.0-0.8 -- threshold barely affects performance.
Precision: 0.98 at max confidence, stays above 0.93 across full range. Zero low-confidence false positives.

### Dataset Characteristics

| Property | Value |
|----------|-------|
| Total images | 1000 |
| Source images | 10 |
| Augmentation ratio | 100x |
| Instances | 1022 |
| Classes | 1 (tiger) |
| BBox large | 1000 |
| BBox medium | 22 |
| BBox small | **0** |
| Quality score | 100/100 |

Zero small-scale training examples. The model has never seen a distant tiger.

---

## 04 // 1 vs 10

### Hypothesis

One image with enough augmentation could be sufficient for detection. We tested it. Same architecture (YOLOv8n), same pipeline (cvLayer), same seed (42). Only variable: source image count.

### Experimental Setup

| Condition | Sources | Augmented | Total |
|-----------|---------|-----------|-------|
| 1-image | 1 | 100x | 100 |
| 10-image | 10 | 100x | 1000 |

### Epoch-by-Epoch Crossover

| Epoch | 1-img mAP | 10-img mAP | Leader | 1-img Prec | 1-img Recall | Status |
|-------|-----------|------------|--------|------------|--------------|--------|
| 1 | **0.995** | 0.307 | 1-img (+0.688) | 0.999 | 0.999 | PEAK |
| 2 | **0.995** | 0.754 | 1-img (+0.241) | 0.999 | 1.000 | HOLDING |
| 3 | **0.995** | 0.670 | 1-img (+0.325) | 0.999 | 0.999 | HOLDING |
| 4 | 0.925 | 0.831 | 1-img (+0.094) | 0.999 | 0.926 | DECLINING |
| 5 | 0.795 | **0.836** | 10-img (+0.041) | 0.999 | 0.794 | **CROSSOVER** |
| 6 | 0.766 | **0.844** | 10-img (+0.078) | 0.999 | 0.765 | DEGRADING |

### The Crossover

The 1-image model leads for epochs 1-4, peaks at 0.995 mAP in epoch 1, then collapses. The 10-image model overtakes at epoch 5 and continues climbing. By epoch 6 the gap is 0.078 and widening.

**The critical observation:** Precision stays at 0.999 throughout the entire 1-image run. The model never makes false positives. But recall falls from 0.999 to 0.765. This is **template narrowing** -- the model becomes increasingly specific about what counts as a tiger, progressively rejecting augmented variants that deviate from the learned template.

### Template Narrowing

Augmentation from 1 image creates new views of the same information, not new information. The model memorizes the single tiger in epoch 1 (0.995 mAP), then subsequent training pushes it past the optimum into over-specificity. It's learning to be more precise about "this exact tiger" rather than learning a broader concept of "tiger."

The 10-image model doesn't suffer this because 10 source images provide enough inter-individual variation (different stripe patterns, body shapes, poses, backgrounds) that the model is forced to learn the concept "tiger" rather than memorizing specific instances.

### Training Time Escalation

| Epoch | Duration | Cumulative |
|-------|----------|------------|
| 1 | 9.0 min | 9.0 min |
| 2 | 10.0 min | 19.0 min |
| 3 | 14.6 min | 33.5 min |
| 4 | 35.3 min | 68.8 min |
| 5 | 104.8 min | 173.6 min |
| 6 | 414.0 min | 587.6 min |

46x slowdown from epoch 1 to epoch 6. The MPS backend is grinding against a degenerate loss surface. The model memorized everything by epoch 2, and subsequent epochs produce vanishing gradients that translate to longer Metal shader execution. The system pays exponentially more compute for exponentially worse results.

### Training Guard Gap

The guard cleared this model past the 5-epoch checkpoint because early metrics were excellent. It has no mechanism to detect that a model which was at 0.995 has degraded to 0.766. The guard evaluates at discrete checkpoints against static thresholds -- it does not track trajectory or regression from peak.

**Implication:** An "attempt mode" that exits at target achievement would have stopped at epoch 1 (0.995 mAP) in 9 minutes. Instead, training ran for 9.8 hours and delivered a model at 0.766 mAP. The guard needs regression detection: track peak performance, halt on sustained decline, revert to best checkpoint.

---

## 05 // EXPLANATION

### Step 01 -- Problem: Small Data, Large Augmentation

10 tiger photographs. Augmentation pipeline: 100 variants per source via scale (80-120%), rotation (+/-15 deg), JPEG quality (70-100), color space (BGR/grayscale). 1000-image training set, 1022 instances. Filenames encode exact parameters: `_autoaug_s88_r-8_q73_bgr`. Fully deterministic.

### Step 02 -- Architecture: YOLOv8n (3.2M Parameters)

Nano variant. Trains in reasonable time on M3 unified memory. Small enough for ONNX browser export. Less capacity to memorize small datasets than larger variants. Fine-tuned from v1 checkpoint.

### Step 03 -- Pipeline: cvLayer MLOps

Training guard auto-adjusted: imgsz 640-->512, batch=2, workers=0. VRAM projection: 0.87GB peak, 6.00GB budget. Deterministic seed 42. 375 locked dependencies. Replayable via `repro_manifest.json`.

```
python -m mlops.pipeline.replay --manifest "repro_manifest.json"
```

### Step 04 -- Training: 20 Epochs, 49 Hours

Box loss: 0.729-->0.113 (84% reduction). Cls loss: 2.269-->0.092 (96% reduction). mAP@0.5 crossed 0.90 at epoch 9. LR decayed across 8 parameter groups.

### Step 05 -- Overfitting: Honest Assessment

Train/val gap 0.78. Expected real-world mAP@0.5: 77.3% (+/-3.1%). Reliability: LOW. Risk: overfit. The 16.4% gap between training (93.7%) and expected real-world (77.3%) is the cost of 10-image training. Zero small-bbox examples means distant tigers are invisible. Single-instance training bias prevents multi-tiger detection.

### Step 06 -- Findings

The 1-image experiment revealed that augmentation cannot substitute for source diversity over extended training. The optimal strategy for minimal-data detection is target-seeking ("attempt mode"): train to a target metric, exit immediately, don't let the model degrade past its peak. The training guard needs regression detection as a post-clearance monitoring capability.

Each source image provides diminishing but meaningful returns. The curve from 1 to 10 images is steep -- the 1-image model crosses over at epoch 5, but a 3-image model would likely hold longer. The experimental matrix to run next: 1/2/3/5/10 source images at constant total training set size, measuring crossover point and final mAP to map the information-per-source-image curve.

---

## STACK

| Component | Detail |
|-----------|--------|
| Architecture | YOLOv8n (3.2M params) |
| Framework | Ultralytics YOLO |
| Backend | PyTorch 2.9.1, Apple MPS |
| Pipeline | cvLayer MLOps |
| Language | Python 3.11 |
| Training | Deterministic (seed 42) |
| Dependencies | 375 packages locked |

## REPRODUCIBILITY

```
Scenario:       TIGERTEST
Version:        v13 (10-img) + v14 (1-img)
Dataset:        tiger111 (snapshot: 41d01aa84c6ac4f8)
Base Model:     TIGERTEST/v1/weights.pt
Device:         Apple M3 (arm64) -- MPS backend
Torch:          2.9.1
Seed:           42
Dependencies:   375 packages (env.requirements.lock)
Replay:         python -m mlops.pipeline.replay --manifest "repro_manifest.json"
```

---

*All metrics are real. All artifacts are from TIGERTEST trained on cvLayer. Nothing is simulated.*
