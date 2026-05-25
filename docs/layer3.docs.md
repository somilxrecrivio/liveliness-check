# Layer 3 · Spatial Passive PAD

> **File:** [`layer3_passive_pad.py`](../layer3_passive_pad.py)
> **Goal:** Differentiate a *live 3-D face* from a *2-D presentation
> attack* (printed photo, displayed screen) using two cheap
> spatial-domain heuristics on the detected face ROI — without any
> training data.

---

## 1. What it does

The layer is a deterministic **IQA-gated three-flag pass/fail** texture
analyser running on the face ROI (or cheek patch). An **image-quality
gate** runs first — if the frame is too dark / bright / flat / blurry /
noisy, the layer refuses to render a verdict and asks for a retake.
Otherwise three spoof flags vote; ANY flag fires → SPOOF.

```
image ──► EXIF transpose ──► MediaPipe BlazeFace
                                     │
                                     ▼
                       ┌────── bbox + 6 keypoints ──────┐
                       │                                 │
                       ▼                                 ▼
              full face crop (256×256)       cheek patch (96×96)
                       │                                 │
                       └──────────── analyse() ──────────┘
                                       │
                       ┌───────────────┴────────────────┐
                       ▼                                ▼
              Uniform LBP (P=8, R=1)          Laplacian (3×3 kernel)
                       │                                │
                       ▼                                ▼
              variance of LBP CODES           variance of Laplacian
              across the ROI                  response
                       │                                │
                       ▼                                ▼
              < 2.0  → "too uniform"          < 60  → "too blurry"
                       │                                │
                       └─────────────── OR ─────────────┘
                                       │
                                       ▼
                                  SPOOF or REAL
```

### 1.1 LBP code variance (the headline texture metric)

Uniform Local Binary Patterns with P=8 neighbours at R=1 pixel. In
this mode skimage produces 10 distinct code values (`0..9`) — the 9
classical uniform patterns (flat, edge, corner, line, …) plus a
"non-uniform" catch-all.

For each pixel of the face ROI we compute its LBP code, then take the
**variance of those code values across the ROI**.

Why this works:

- **Real skin** has many distinct micro-textures within the same ROI:
  smooth patches between pores, edges around each pore, corners
  around follicles, lines around fine wrinkles. The pixels are
  distributed across several different LBP codes → **high code
  variance** (typically 2.5–5.0 on a real face crop).
- **Printed photo** halftones smooth the micro-texture during ink
  rendering. Most pixels classify as the same "uniform smooth" LBP
  code → **low code variance** (typically 0.5–1.5).
- **Screen replay** does the same via subpixel interpolation, plus
  lossy compression on the camera side.

### 1.2 LBP code variance vs LBP histogram-bin variance

These are *different* metrics with **opposite directions**:

| Metric | What it measures | Real skin | Flat spoof |
|---|---|---|---|
| **Code variance** (used here) | spread of LBP codes across pixels | **HIGH** | LOW |
| Histogram-bin variance | how peaked the code distribution is | LOW | HIGH (all mass in one bin) |

The first iteration of this layer used the wrong one. The wording in
the original spec — *"pores have high variance, digital pixels/prints
do not"* — refers to the code variance, not the histogram-bin
variance, because:

- A perfectly flat image gives **every pixel the same LBP code** → 1
  bin gets 100 % of the histogram mass → histogram-bin variance is
  maximised (its highest possible value), but **code variance is 0**.

Smoke-testing on synthetic stimuli is what flushed this out — the
"flat patch" produced the *highest* histogram-bin variance, which is
the opposite of the desired direction.

### 1.3 IQA quality gate (NEW)

Four cheap no-reference image-quality stats computed on the same ROI;
the verdict is **withheld** (warning banner, no spoof claim) when any
of them falls out of bounds:

| Stat | Default bound | What it catches |
|---|---|---|
| brightness (mean) | 40 ≤ x ≤ 235 | too dark / blown-out frames |
| contrast (std) | ≥ 18 | flat lighting / overexposed |
| dynamic range (p95−p5) | ≥ 50 | low-bit-depth / dim frames |
| noise estimate (std of gray − median3) | ≤ 22 | motion blur / sensor noise |

This protects the downstream spoof verdict from being polluted by
captures the upstream pipeline shouldn't have accepted in the first
place. A bogus "SPOOF" verdict on a too-dark frame is worse than no
verdict — it teaches the operator to ignore the layer.

### 1.4 Laplacian variance (sharpness / focus)

The classical Pech-Pacheco focus measure (Pech-Pacheco et al. 2000).
Convolve the grayscale ROI with the 3×3 Laplacian kernel; the variance
of the response is a good proxy for high-frequency content. Sharp
in-focus skin has many strong Laplacian responses around pores and
edges → high variance. Defocused images, screen anti-aliasing, and
print blur all suppress those responses → low variance.

Typical ranges:

- Sharp face crop: **100–1000+**
- Mildly defocused: **50–100**
- Screen replay (defocus + screen blur): **20–80**
- Heavily blurred / out of focus: **< 20**

### 1.5 Moiré peak detection (NEW)

A 2-D FFT-based check ported from the original slope+peak Layer 2.
Pipeline: Hann-windowed FFT of the ROI → log-magnitude → annulus
`0.25 < r/R < 0.85` → count pixels exceeding `mean + 3.5σ` inside
that band.

| Source | Typical peak count |
|---|---|
| Real skin | 0–5 |
| Printed halftone | 50–1000+ |
| Screen replay with visible Moiré | 20–100+ |

Real face crops have no peaked structure in this band; printed
halftones produce a regular dot grid that lights up the annulus.
This was the strongest detector of paper attacks in the old Layer 2;
Layer 2 is now the YOLO+FFT-variance hybrid, so the explicit peak
counter moved here.

### 1.6 Why MediaPipe instead of Haar

Layer 2 demonstrated Haar's fragility — it failed on EXIF-rotated
phone photos, glasses-wearing subjects, and side profiles. We switched
to MediaPipe Tasks API (BlazeFace short-range backbone) here for:

- Robust to face tilt, occlusion, and lighting.
- Returns 6 named keypoints (eyes, nose tip, mouth, ear tragions),
  which lets us extract a **cheek patch** — the cleanest skin ROI in
  the frame, free of glasses / beard / eyebrows / mouth / background.
- The `.tflite` model bundle is auto-downloaded (~230 KB) on first
  use and cached to `./models/blaze_face_short_range.tflite`.

### 1.7 Cheek patch ROI (recommended)

When `Use cheek patch` is enabled (default), the layer extracts a
square patch midway between the eye and mouth keypoints on one cheek.
This is the most discriminative skin region: solid cheek skin with no
contaminating features. All metrics improve substantially when run on
the cheek vs the full face crop, especially for subjects with
glasses, beards, or busy backgrounds inside the bbox.

If the geometry degenerates (face too small / too sideways), the
layer falls back to the full face crop with a warning.

---

## 2. Pros

- **No training data needed.** Pure linear-algebra heuristics. Works
  on any subject, any pose.
- **Two independent physical signals.** LBP measures
  *micro-texture richness*; Laplacian measures *high-frequency
  sharpness*. Either alone catches a different attack mode (LBP for
  print/screen, Laplacian for defocus/blur).
- **Cheap.** Face detection ~5 ms + LBP ~3 ms + Laplacian ~1 ms.
  Real-time on CPU.
- **Explainable.** Both metrics are single numbers with clear
  physical interpretation. The LBP code map and `|Laplacian|`
  response are shown so the operator can see the evidence.
- **MediaPipe handles tilt and occlusion** that Haar misses.
- **Cheek patch isolates skin** — no glasses, beard, eyebrows, or
  background contaminating the measurement.
- **Honest no-face behaviour.** If MediaPipe can't find a face, the
  layer refuses to render a verdict (no centre-crop fallback). This
  was a lesson learned from Layer 2's debugging.

## 3. Cons & failure modes

- **Single-feature LBP is borderline-discriminative on its own.**
  Real-skin vs. high-quality-print code-variance ranges overlap; the
  classical PAD literature gets ~75–85 % accuracy from LBP alone.
  Production should learn a classifier over LBP + Laplacian +
  additional features (HOG, GLCM, colour stats).
- **Thresholds are content-dependent.** Subjects with smooth skin
  (clean-shaven, makeup) drop closer to the threshold; subjects with
  stubble or pores ride well above it.
- **Laplacian is sensitive to camera focus.** A real face captured
  out of focus will fail the sharpness flag — but that's arguably
  correct behaviour for a passive PAD system (a too-blurry frame
  isn't reliable evidence either way).
- **JPEG compression at low quality flattens both metrics.** Below
  ~Q=70 the file's HF content is heavily quantised; both LBP and
  Laplacian variances drop into the spoof range.
- **Modern phone ISPs apply aggressive denoising** that suppresses
  skin micro-texture before saving. Phone-captured real faces can
  show LBP code variance around 2.0–2.5, right at the threshold.
- **Defeated by high-resolution print or 4K screen** at close range
  with matched focus. The signal is texture quality, and a
  high-quality reproduction of skin texture is hard to distinguish
  from skin.
- **MediaPipe occasionally hallucinates a face** on cluttered
  backgrounds (rare, but happens). The verdict on those crops is
  meaningless.

---

## 4. Alternatives & complementary signals

### 4.1 Multi-scale LBP + classifier
LBP at multiple radii (P=8,R=1 + P=16,R=2 + P=24,R=3) concatenated
into a feature vector, then SVM/GBM classifier trained on
real/spoof pairs.
- **Pros:** Captures texture at multiple scales; the classical PAD
  baseline (Maatta et al. 2011).
- **Cons:** Requires labelled training data; domain-shift sensitive.

### 4.2 Colour LBP / chromatic texture
Compute LBP on each colour channel or in HSV/YCbCr, concatenate
histograms.
- **Pros:** Captures colour-texture coupling that grayscale LBP
  misses; prints often have characteristic colour cast.
- **Cons:** Larger feature vector; sensitive to white balance.

### 4.3 Gray-Level Co-occurrence Matrices (GLCM)
Texture statistics from co-occurrence matrices: contrast,
homogeneity, energy, correlation.
- **Pros:** Complementary to LBP; classical and well-studied.
- **Cons:** Slower to compute; thresholds also need calibration.

### 4.4 Deep texture features (e.g. CDCN, DeepPixBiS)
End-to-end neural PAD models.
- **Pros:** SOTA on CelebA-Spoof, OULU-NPU benchmarks.
- **Cons:** Heavy, opaque, requires labelled training data,
  domain-shift sensitive.

### 4.5 rPPG (remote photoplethysmography)
Recover subtle skin-colour pulsation across video frames.
- **Pros:** Proves *liveness* — a printed photo has no pulse signal,
  full stop. Strongest passive signal on the bench.
- **Cons:** Requires stable video, good lighting, ~10 s window;
  ill-suited to one-shot verification.

### 4.6 Active liveness challenges
Prompt the user to blink, smile, or turn their head.
- **Pros:** Different attack surface, catches replay attacks.
- **Cons:** Worse UX, slower onboarding, defeated by real-time
  deepfake puppeteering.

### 4.7 Depth / 3D liveness sensors
Structured-light or ToF sensors (FaceID).
- **Pros:** Cryptographically strong against any 2-D attack.
- **Cons:** Hardware-locked; not portable.

---

## 5. How this layer fits into the full stack

| Layer | Catches                                  | Cost           |
| ----- | ---------------------------------------- | -------------- |
| 1     | Injection (virtual cams, file replay)    | µs/frame       |
| 2     | Deepfake / screen / print (YOLO + FFT)   | ~30 ms/frame   |
| **3** | **Print / screen physical texture (LBP + Laplacian)** | **~10 ms/frame** |
| 4     | Identity match (InsightFace embeddings)  | ~100 ms        |

Layer 3 is an independent texture-domain check that **doesn't share
signal sources with Layer 2**. Layer 2's YOLO + FFT both look at the
2-D image content directly; Layer 3 looks at the spatial *texture
statistics* of the same content. An attacker who beats one approach
(say, a high-quality deepfake that fools YOLO and has a clean
spectrum) is unlikely to also produce skin micro-texture that matches
real pores at the LBP code-variance level.

---

## 6. Testing recipes

1. **Webcam → your face** → expect REAL. Both flags green.
   LBP code variance 2.5–5.0; Laplacian variance 100–1000.

2. **Print a face on plain paper, hold it up to the webcam** →
   expect SPOOF (LBP flag). Halftone collapses LBP code variance
   into the 0.5–1.5 range.

3. **Show a face on your phone or monitor, photograph with the
   webcam** → expect SPOOF (LBP flag, often + Laplacian flag).
   Screen subpixel interpolation + camera defocus on the screen
   surface flatten both metrics.

4. **Deliberately defocus the webcam on a real face** → expect
   SPOOF (Laplacian flag). Real subject, but the captured frame
   is too low-quality to verify.

5. **Stand 2 m from the webcam** (small face in frame) → expect
   REAL but the cheek patch may degenerate; the layer falls back
   to the full face crop with a warning.

If a real subject is misclassified, look first at:
- The **cheek patch image** — is it actually skin, or did the
  geometry slip onto hair / glasses / beard?
- The **LBP false-colour image** — does it look varied (real) or
  uniform (spoof)?
- The **|Laplacian| response** — are there bright spots around pores
  and edges, or is it dim?

---

## 7. Recommendations for production

1. **Always run on the cheek patch**, never the full face. The
   discriminative gap between real and spoof is 2–3× wider on a
   clean skin ROI than on a bbox that includes eyes, mouth, glasses,
   and background.
2. **Calibrate thresholds per device class.** Phone ISPs that
   aggressively denoise skin produce systematically lower LBP code
   variance — recalibrate the 1st-percentile threshold on labelled
   captures per browser / OS / sensor.
3. **Use LBP + Laplacian + colour + HOG as a feature vector** for a
   trained classifier rather than two independent thresholds. A
   shallow GBM or logistic regression generalises far better than
   ANDed hand-tuned thresholds.
4. **Combine with Layer 2 (YOLO + FFT) and Layer 4 (identity).** A
   single anti-spoofing signal is never sufficient for a high-stakes
   biometric decision.
5. **Reject frames with Laplacian variance < 20** unconditionally
   (image too blurry to assess regardless of spoofing) — ask the
   user to retry.
6. **Log the cheek patch, LBP code map, and Laplacian response**
   alongside the verdict so false positives / negatives are
   triagable post-hoc.
