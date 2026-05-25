# Liveliness Check

A modular **Presentation Attack Detection (PAD)** and **Injection Attack
Detection (IAD)** prototype suite for face-based identity verification.
Each defence layer is built as a **standalone Streamlit app** so the
underlying math and ML models can be tested in isolation before being
fused into a production pipeline.

The goal is **defense in depth**: no single signal is sufficient
against a determined attacker, but the failure modes of these six
layers are largely independent. An attacker who beats one layer is
unlikely to beat them all.

---

## What this project covers

The system is built to detect and reject the full range of common
face-verification attacks:

| Attack class | Example | Caught by |
|---|---|---|
| **Injection** | Virtual camera replaying a video file | Layer 1 (frame-pacing variance) |
| **Deepfake / GAN-generated faces** | StyleGAN, diffusion-model portraits | Layer 2 (YOLO + FFT) |
| **Screen replay (Moiré)** | Phone showing a face, photographed | Layer 3 (Moiré peaks), Layer 5 (depth), Layer 6 (pulse) |
| **Printed-photo attack** | Paper print of a face on a stand | Layer 3 (LBP, Moiré), Layer 5 (depth), Layer 6 (motion + pulse) |
| **Defocused / blurred capture** | Poor-quality frames that mask spoofs | Layer 3 (Laplacian + IQA gate) |
| **Static-image attack** | A pristine still image presented to the camera | Layer 6 (motion variance, pulse SNR) |
| **Looped video replay** | A 5 s clip of the victim played on repeat | Layer 6 (motion-displacement periodicity) |
| **Identity substitution** | A real person impersonating the ID owner | Layer 4 (ArcFace cosine similarity) |
| **Low-quality / unusable frame** | Too dark / blown out / motion-blurred | Layer 3 IQA quality gate (refuses verdict) |

### The six passive-check signals

| # | Signal | Layer | What it measures |
|---|---|---|---|
| 1 | **Texture analysis** | Layer 3 | LBP code variance on the cheek patch — real skin has varied micro-texture; prints/screens collapse onto one "uniform smooth" code |
| 2 | **Moiré pattern detection** | Layer 3 | FFT annulus peak counter — printed halftones and screen-camera beat frequencies produce distinct HF peaks that real skin doesn't |
| 3 | **Monocular depth estimation** | Layer 5 | Depth Anything V2 Small — real faces have nose-vs-cheek-vs-corner depth structure; 2-D presentations are flat |
| 4 | **rPPG (heart-rate pulse)** | Layer 6 | CHROM algorithm on cheek-RGB time series — recovers the pulse signal that only exists on living tissue |
| 5 | **Micro-movement analysis** | Layer 6 | Frame-to-frame keypoint displacement + autocorrelation — distinguishes static photos, real subjects, and looped replays |
| 6 | **Image quality features** | Layer 3 | Brightness, contrast, dynamic range, noise, Laplacian sharpness — quality gate that refuses verdicts on unusable frames |

---

## Full defence-in-depth pipeline

```
                              webcam / uploaded image
                                       │
                                       ▼
            ┌──────────────────────────────────────────────────────────┐
            │  Layer 1 · Hardware & Metadata IAD                       │
            │  Inter-frame timestamp δ variance                        │
            │  Bot stream → σ²(δ) ≈ 0 → REJECT                         │
            │  Cost: µs/frame                                          │
            └──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
            ┌──────────────────────────────────────────────────────────┐
            │  Layer 2 · Deepfake IAD (Hybrid YOLO + FFT)              │
            │  YOLO real/fake classifier ∨ FFT variance > threshold    │
            │  → REJECT                                                │
            │  Cost: ~30 ms/frame                                      │
            └──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
            ┌──────────────────────────────────────────────────────────┐
            │  Layer 3 · Spatial Passive PAD                           │
            │  IQA gate ▸ LBP code var ∨ Laplacian ∨ Moiré peaks       │
            │  → REJECT                                                │
            │  Cost: ~10 ms/frame                                      │
            └──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
            ┌──────────────────────────────────────────────────────────┐
            │  Layer 5 · Monocular Depth PAD                           │
            │  Depth variance ∨ centre-minus-edge depth                │
            │  → REJECT                                                │
            │  Cost: ~200 ms/face (CPU)                                │
            └──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
            ┌──────────────────────────────────────────────────────────┐
            │  Layer 6 · Video Liveness (rPPG + Micro-movement)        │
            │  CHROM pulse SNR ∨ motion variance ∨ periodicity         │
            │  → REJECT                                                │
            │  Cost: ~10 s capture + ~1 s analysis                     │
            └──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
            ┌──────────────────────────────────────────────────────────┐
            │  Layer 4 · Biometric Matching                            │
            │  InsightFace ArcFace · cosine similarity ≥ threshold     │
            │  → MATCH or NO MATCH                                     │
            │  Cost: ~100 ms/face                                      │
            └──────────────────────────────────────────────────────────┘
```

**Layer 4 deliberately runs last.** Identity matching alone cannot tell a
face from a photograph of that face — it gives a printed spoof full
credit. The PAD layers must execute first.

---

## Tech stack

| Concern | Libraries |
|---|---|
| **UI** | `streamlit` |
| **Computer vision** | `opencv-python`, `opencv-contrib-python`, `scikit-image`, `Pillow`, `cvzone` |
| **Math / signal processing** | `numpy`, `pandas`, `scipy` (FFT, Butterworth filter, autocorrelation) |
| **Face detection** | `mediapipe` (BlazeFace via Tasks API), OpenCV Haar (Layer 2) |
| **Deep models** | `ultralytics` (YOLO), `insightface` (RetinaFace + ArcFace), `onnxruntime` (Depth Anything V2 Small) |

### Models used

| Model | Where | Format | Size | Source |
|---|---|---|---|---|
| **YOLO real/fake classifier** | Layer 2 | `best.pt` (user-supplied) | varies | trained offline, dropped in project root |
| **BlazeFace short-range** | Layers 3 / 5 / 6 | `.tflite` | ~230 KB | auto-downloaded from `storage.googleapis.com/mediapipe-models/` |
| **InsightFace `buffalo_l`** | Layer 4 | 5 × `.onnx` (RetinaFace + 2D / 3D landmark + ArcFace W600K-R50 + age/gender) | ~341 MB total | auto-downloaded to `~/.insightface/models/` |
| **Depth Anything V2 Small** | Layer 5 | `.onnx` | ~99 MB | auto-downloaded from HuggingFace `onnx-community/depth-anything-v2-small` |

All models other than `best.pt` are fetched on first run and cached.

---

## Setup

```bash
cd /Users/somilgupta/Desktop/recrivio/liveliness-check

# Create a Python 3.11 venv (recommended — MediaPipe and InsightFace
# have the cleanest wheel coverage at 3.11; later versions can break).
python3.11 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt

# Layer 2 needs a trained YOLO classifier. Place it at:
#   ./best.pt
# (a CelebA-Spoof-trained anti-spoof YOLO, or any 2-class real/fake model)
```

First-run model downloads land in `./models/` (BlazeFace, Depth Anything)
and `~/.insightface/models/buffalo_l/` (InsightFace).

---

## Running individual layers

Each layer is a standalone Streamlit app:

```bash
.venv/bin/streamlit run layer1_metadata.py        # Frame timing IAD
.venv/bin/streamlit run layer2_deepfake.py        # YOLO + FFT
.venv/bin/streamlit run layer3_passive_pad.py     # LBP + Laplacian + Moiré + IQA
.venv/bin/streamlit run layer4_biometrics.py      # ArcFace identity matching
.venv/bin/streamlit run layer5_depth_pad.py       # Monocular depth PAD
.venv/bin/streamlit run layer6_video_liveness.py  # rPPG + micro-movement
```

---

## Layer 1 · Hardware & Metadata IAD

**File:** [`layer1_metadata.py`](layer1_metadata.py) ·
**Docs:** [`docs/layer1.docs.md`](docs/layer1.docs.md)

Detects video injection attacks (virtual cameras, OBS Virtual Cam,
`v4l2loopback`, `ffmpeg` piped through a fake device) by analysing the
**variance of inter-frame timestamp deltas**.

A real camera accumulates jitter from many independent sources (OS
scheduler, sensor readout, USB bus contention, ISP timing). A bot
replaying a pre-rendered file emits frames on a mathematically perfect
cadence. The variance separates them.

```
                       captured frame timestamps
                       t₀ < t₁ < t₂ < … < tₙ
                                │
                                ▼
                       δᵢ = tᵢ₊₁ − tᵢ          (inter-frame deltas)
                                │
                                ▼
                       σ²(δ) = Var({δᵢ})
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
            σ² < threshold             σ² ≥ threshold
            INJECTION SUSPECTED        LIKELY REAL CAMERA
```

**Three frame sources** in the UI: simulated bot (perfect cadence),
simulated real camera (Gaussian jitter), and **live webcam capture**
via `cv2.VideoCapture` for end-to-end validation against OBS Virtual
Camera.

**Tech:** `numpy` · `opencv-python` · `streamlit`

---

## Layer 2 · Deepfake IAD (YOLO + FFT Hybrid)

**File:** [`layer2_deepfake.py`](layer2_deepfake.py) ·
**Docs:** [`docs/layer2.docs.md`](docs/layer2.docs.md)

A two-detector ensemble that votes on every face in the frame:

```
                              frame
                                │
                                ▼
                ┌──────────────────────────────────┐
                │  YOLO (best.pt)                  │
                │  bbox + class ∈ {real, fake}     │
                │  confidence ≥ 0.50               │
                └────────────┬─────────────────────┘
                             │
                             ▼
                       face crop
                             │
                             ▼
                ┌──────────────────────────────────┐
                │  FFT variance check              │
                │  variance of 20·log|F(face_crop)|│
                │  > FFT_VARIANCE_THRESHOLD (1500) │
                └────────────┬─────────────────────┘
                             │
                             ▼
                ┌──────────────────────────────────┐
                │  OR ensemble:                    │
                │  YOLO=fake ∨ FFT=fake → FAKE     │
                │  else → REAL                     │
                └──────────────────────────────────┘
```

The fail-safe OR rule biases toward **false positives over false
negatives** — it's safer to make a real user re-shoot than to admit a
spoofed identity. Live preview overlays both individual votes on the
detection box so an operator can see which detector triggered.

**Tech:** `ultralytics` (YOLO) · `opencv-python` (FFT, video) ·
`cvzone` (overlays) · `streamlit`

---

## Layer 3 · Spatial Passive PAD

**File:** [`layer3_passive_pad.py`](layer3_passive_pad.py) ·
**Docs:** [`docs/layer3.docs.md`](docs/layer3.docs.md)

Three independent texture-domain flags gated by an image-quality check:

```
image ─► EXIF transpose ─► resize ≤1024 ─► MediaPipe BlazeFace
                                                  │
                                                  ▼
                              face bbox + 6 keypoints
                                                  │
                                                  ▼
                                  extract cheek patch
                                  (midpoint of eye and mouth)
                                                  │
                  ┌───────────────────────────────┴───────────────────────────┐
                  ▼                                                           ▼
        IQA quality gate                                          three spoof flags
        brightness, contrast,                          LBP code variance < threshold
        dynamic range, noise                           Laplacian variance < threshold
                  │                                    Moiré peak count > threshold
            out of range?                                          │
                  │                                                ▼
                  ▼                                       ANY flag → SPOOF
            WITHHOLD VERDICT
            (ask for retake)
```

The IQA gate **refuses to render a verdict** on out-of-range frames
(too dark, too bright, flat, noisy) rather than producing a bogus
spoof verdict on garbage input.

The three spoof signals attack different physics:
- **LBP code variance** — real skin has varied micro-textures (smooth
  patches, pores, follicles) → many distinct LBP codes → high
  variance. Prints/screens collapse onto one "uniform smooth" code.
- **Laplacian variance** — Pech-Pacheco focus measure. Sharp real
  skin → high variance; blurred / defocused / screen-replayed → low.
- **Moiré peak count** — FFT annulus pixel count above 3.5σ.
  Printed halftones and screen-camera beat frequencies produce
  visible peaks; real skin doesn't.

**Tech:** `mediapipe` (BlazeFace) · `scikit-image` (LBP) ·
`opencv-python` (Laplacian, FFT, colormap) · `Pillow` · `streamlit`

---

## Layer 4 · Biometric Matching

**File:** [`layer4_biometrics.py`](layer4_biometrics.py) ·
**Docs:** [`docs/layer4.docs.md`](docs/layer4.docs.md)

Once the PAD layers have confirmed the input is a live human, this
layer decides whether that human is **the same person** as the stored
reference ID.

```
reference ID                              probe (live capture)
       │                                            │
       ▼                                            ▼
  EXIF transpose                            EXIF transpose
       │                                            │
       ▼                                            ▼
  InsightFace FaceAnalysis                  InsightFace FaceAnalysis
   (RetinaFace + alignment +                 (same pipeline)
    ArcFace W600K-R50)
       │                                            │
       ▼                                            ▼
  embedding_ref (512-D, L2-norm)            embedding_probe
                  │                                 │
                  └─────── dot product ─────────────┘
                                  │
                                  ▼
                       cosine similarity ∈ [−1, +1]
                                  │
                                  ▼
                      sim ≥ threshold (default 0.40)
                              ↓               ↓
                            MATCH         NO MATCH
```

Threshold semantics:

| Threshold | Character | Use case |
|---|---|---|
| 0.30 | very loose | risk-tolerant onboarding |
| **0.40** | **balanced** | **default — KYC, login** |
| 0.50 | strict | high-stakes auth, financial actions |
| 0.60 | very strict | duplicate detection across a DB |

**Critical security note:** this layer **cannot tell a real face from
a photograph of that face** — both produce nearly identical embeddings.
A printed spoof of the reference will score ~0.95 cosine similarity
and pass. Layer 4 *must* be gated by the PAD layers (1, 2, 3, 5, 6).
Running it alone defeats the entire security model.

**Tech:** `insightface` · `onnxruntime` · `opencv-python` ·
`Pillow` · `streamlit`

---

## Layer 5 · Monocular Depth PAD

**File:** [`layer5_depth_pad.py`](layer5_depth_pad.py) ·
**Docs:** [`docs/layer5.docs.md`](docs/layer5.docs.md)

A real face has visible 3-D structure — the nose protrudes 2–4 cm
beyond the cheeks, which are themselves closer than the ears /
background. A 2-D presentation attack (printed photo, screen) is
flat — every pixel of the depicted face sits at the same physical
depth.

```
image ─► MediaPipe BlazeFace ─► face bbox
                                      │
                                      ▼
                    crop with 15% margin (corners
                    include cheek / hair / background)
                                      │
                                      ▼
                  Depth Anything V2 Small (518×518 ONNX, CPU)
                                      │
                                      ▼
                       inverse-depth map → normalise [0, 1]
                                      │
                  ┌───────────────────┴─────────────────────────┐
                  ▼                                             ▼
        variance(depth)                       centre(30%) − edge(4 corners × 15%)
                  │                                             │
                  ▼                                             ▼
            < 0.030 → FLAG                                 < 0.10 → FLAG
                  │                                             │
                  └───────────────────  OR  ────────────────────┘
                                      ▼
                                  SPOOF or REAL
```

Why **both flags**:
- **Flat homogeneous photo** fails the variance flag but might pass
  centre-minus-edge by chance.
- **Photo of a photo at an angle** can have non-trivial variance from
  perspective rendering, but centre-vs-edge is still wrong.

Either alone misses one mode; together they cover both.

**Tech:** `onnxruntime` (Depth Anything V2 Small) · `mediapipe`
(BlazeFace) · `opencv-python` · `Pillow` · `streamlit`

---

## Layer 6 · Video Liveness (rPPG + Micro-movement)

**File:** [`layer6_video_liveness.py`](layer6_video_liveness.py) ·
**Docs:** [`docs/layer6.docs.md`](docs/layer6.docs.md)

The **first multi-frame layer**. Two physical signals that only exist
across multiple frames of a truly living subject:

1. **Pulse (rPPG)** — sub-percent green-band absorbance changes from
   cardiovascular oxygenation produce a 0.7–4 Hz (42–240 bpm) signal
   on cheek skin.
2. **Involuntary micro-movement** — breathing, microsaccades, and
   skin micro-flexing translate the face by tenths of a pixel
   irregularly. Static photos have zero displacement; looped video
   replays have *periodic* displacement.

```
[Start Capture button]
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  cv2.VideoCapture loop @ ~30 fps for ~10 s                  │
│    per-frame:                                                │
│      MediaPipe BlazeFace  →  bbox + 6 keypoints              │
│      extract cheek patch  →  mean RGB (3 floats)             │
│      store: timestamp, bbox, keypoints, cheek_RGB_mean       │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐    ┌────────────────────────────┐
│ CHROM rPPG                   │    │ Motion analysis            │
│  X = 3R − 2G                 │    │  Δ_t = ||(nose+mouth)_t    │
│  Y = 1.5R + G − 1.5B         │    │            − ...t−1 ||     │
│  bandpass [0.7, 4.0] Hz      │    │  var(Δ) → motion_variance  │
│  pulse = X_bp − αY_bp        │    │  ACF(Δ) → periodicity      │
│  FFT → peak / median = SNR   │    │   (max |ACF| lag > 0.5 s)  │
└──────────────────────────────┘    └────────────────────────────┘
        │                                            │
   SNR < threshold                  var < min  OR  periodicity > max
        │                                            │
        └─────────────────── OR ─────────────────────┘
                              ▼
                         SPOOF or REAL
```

Three flag interpretations:
- **rPPG SNR low** → no detectable pulse → static photo / non-living.
- **Motion variance low** → face too still → static photo on a stand.
- **Periodicity high** → motion repeats with period > 0.5 s →
  **looped video replay attack**.

**Tech:** `opencv-python` (VideoCapture) · `mediapipe` (BlazeFace) ·
`scipy.signal` (Butterworth bandpass, autocorrelation) · `numpy` ·
`pandas` · `streamlit`

---

## Project structure

```
liveliness-check/
├── README.md                       ← you are here
├── requirements.txt                ← pinned dependency list
├── best.pt                         ← user-supplied YOLO weights (Layer 2)
│
├── layer1_metadata.py              ← Frame-timing IAD
├── layer2_deepfake.py              ← YOLO + FFT hybrid
├── layer3_passive_pad.py           ← LBP + Laplacian + Moiré + IQA
├── layer4_biometrics.py            ← ArcFace identity matching
├── layer5_depth_pad.py             ← Depth Anything V2 monocular depth
├── layer6_video_liveness.py        ← CHROM rPPG + motion analysis
│
├── docs/
│   ├── layer1.docs.md              ← per-layer deep dives
│   ├── layer2.docs.md
│   ├── layer3.docs.md
│   ├── layer4.docs.md
│   ├── layer5.docs.md
│   └── layer6.docs.md
│
├── models/                         ← auto-cached on first run
│   ├── blaze_face_short_range.tflite   (~230 KB · Layers 3, 5, 6)
│   └── depth_anything_v2_small.onnx    (~99 MB · Layer 5)
│
└── .venv/                          ← Python 3.11 venv (gitignored)
```

The InsightFace `buffalo_l` bundle (Layer 4) is cached at
`~/.insightface/models/buffalo_l/` (~341 MB across 5 ONNX models),
managed by the InsightFace library itself.

---

## Threat-model coverage

The six layers cover **largely orthogonal attack surfaces**, so an
attacker who beats one is unlikely to beat them all simultaneously:

| Attack | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|
| OBS virtual camera replay | **✓** | partial | — | — | — | — |
| Printed photo presented to camera | — | partial | **✓** | — | **✓** | **✓** |
| 4K screen replay | — | partial | partial | — | **✓** | **✓** |
| GAN / diffusion deepfake (still) | — | **✓** | partial | — | partial | **✓** |
| Deepfake video (cooperative motion) | — | **✓** | partial | — | partial | **✓** (no pulse) |
| Looped video replay | — | — | — | — | — | **✓** (periodicity) |
| Different real person impersonating | — | — | — | **✓** | — | — |
| Out-of-focus / unusable frame | — | — | **✓** (IQA gate) | — | — | — |

**Recommended production ordering** (cheap-to-expensive, reject early):

1. **Layer 1** — cheap injection check
2. **Layer 3** — IQA gate + cheap texture / Moiré
3. **Layer 2** — YOLO + FFT
4. **Layer 5** — depth (for borderline cases)
5. **Layer 6** — pulse + motion (enrolment / re-auth / borderline)
6. **Layer 4** — identity match (only after liveness is confirmed)

---

## Per-layer documentation

For algorithmic detail, threshold rationale, alternative approaches,
known failure modes, and production recommendations:

- [`docs/layer1.docs.md`](docs/layer1.docs.md) — frame-timing variance
- [`docs/layer2.docs.md`](docs/layer2.docs.md) — YOLO + FFT ensemble
- [`docs/layer3.docs.md`](docs/layer3.docs.md) — LBP + Laplacian + Moiré + IQA
- [`docs/layer4.docs.md`](docs/layer4.docs.md) — ArcFace embedding match
- [`docs/layer5.docs.md`](docs/layer5.docs.md) — monocular depth PAD
- [`docs/layer6.docs.md`](docs/layer6.docs.md) — rPPG + micro-movement

Each per-layer doc follows the same template: **what it does · pros ·
cons · alternatives · stack position · testing recipes · production
recommendations**.
