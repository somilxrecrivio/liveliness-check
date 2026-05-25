# Layer 2 · Hybrid Detection (YOLO + FFT)

> **File:** [`layer2_deepfake.py`](../layer2_deepfake.py)
> **Goal:** Detect spoofed faces (printed photos, screen replay, deepfakes)
> by combining a **pre-trained YOLO real/fake classifier** with a
> deterministic **FFT variance** sanity check, fused via an **OR**
> ensemble rule.

---

## 1. What it does

The layer is an **ensemble of two heterogeneous detectors** that vote on
every face in the frame:

```
                    ┌──────────────────────────────────┐
                    │  YOLO model (best.pt)            │
       frame ─────► │  bbox + class ∈ {real, fake} +   │
                    │  conf (threshold 0.50)           │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                          ┌────────────────┐
                          │  face crop     │
                          └────────┬───────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  FFT variance check              │
                    │  variance of 20·log|F|           │
                    │  > 1500  →  "fake"               │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  OR ensemble (fail-safe):        │
                    │  YOLO=fake OR FFT=fake → FAKE    │
                    │  else → REAL                     │
                    └──────────────────────────────────┘
```

Both modes (live webcam and static upload) run the same pipeline; only
the input source differs.

### 1.1 YOLO real/fake classifier

- Loaded from `best.pt` in the working directory via
  `ultralytics.YOLO` (`@st.cache_resource` so the model loads once).
- Returns detection objects with `xyxy` bbox, class index, and a
  confidence score.
- `CLASS_NAMES = ["real", "fake"]`. The order matters — if your model
  was trained with the indices swapped, flip this list.
- Detections below `CONFIDENCE_THRESHOLD = 0.50` are ignored.

### 1.2 FFT variance sanity check

`calculate_fft_variance(face_crop)`:

1. Convert the YOLO-cropped face to grayscale.
2. `cv2.dft` → `np.fft.fftshift` → 20·log|F| magnitude spectrum.
3. Return `np.var(magnitude_spectrum)`.

The default `FFT_VARIANCE_THRESHOLD = 1500` is in the units produced by
the `20·log10(...)` scaling (much larger numbers than a `log1p`-based
spectrum would give). Above that threshold the math vote is "fake".

> **Calibration warning** — this threshold is hardware-dependent. A
> 720p webcam, a 1080p webcam, a phone camera, and a DSLR will all
> produce different variance ranges for the same scene. You will need
> to tune `FFT_VARIANCE_THRESHOLD` empirically per device.

### 1.3 OR ensemble (fail-safe voting)

```python
if yolo_guess == "fake" or math_guess == "fake":
    final_label = "FAKE"
else:
    final_label = "REAL"
```

A single "fake" vote from either detector wins. This biases the layer
toward **false positives over false negatives** — it's safer to make
a real user re-shoot than to admit a spoofed identity. The diagnostic
text on screen always shows both individual votes
(`YOLO:R | FFT:1234`) so the operator can see which detector
triggered.

---

## 2. Pros of the hybrid approach

- **Two completely different failure modes** — YOLO can fail on a
  novel attack it wasn't trained on; FFT can fail on a high-quality
  spoof that passes the variance gate. Both failing at the same time
  is unusual.
- **YOLO catches deepfakes the math doesn't**. A modern diffusion
  model can produce a spectrum that looks natural, but the YOLO
  classifier (if trained on diffusion samples) still flags it.
- **FFT catches attacks YOLO doesn't generalise to**. If an attacker
  presents a print/screen that the YOLO training set never saw, the
  Moiré/halftone variance still spikes.
- **Real-time capable**. YOLO Nano-class models run 30+ fps on CPU;
  FFT on a face crop is ~1 ms.
- **Live preview with cvzone overlays** — the corner-rect drawing and
  diagnostic text make the dual decision instantly readable while
  testing.
- **No multi-step interaction** — single inference call per frame, so
  it slots into a real-time pipeline.

## 3. Cons & failure modes

- **Requires a pre-trained `best.pt`**. Without that model file the app
  exits at startup. The detector is only as good as its training set —
  attacks not represented at training time will silently pass YOLO.
- **YOLO is opaque**. You can't audit *why* it called something fake;
  the confidence score is correlated with calibration but not causal.
- **Class index ambiguity**. If your YOLO model's class indices are
  swapped (`["fake", "real"]` instead of `["real", "fake"]`), every
  prediction inverts. There's no automatic detection — you have to
  inspect a few outputs and flip `CLASS_NAMES` if needed.
- **FFT threshold is hardware-coupled**. `1500` is *not* a universal
  constant. On a high-resolution sensor the variance baseline rises;
  on a webcam after JPEG transcoding it drops. Calibration is
  mandatory.
- **OR ensemble inflates false positives**. Either detector being
  wrong → user re-shoot. Acceptable in onboarding flows, painful in
  continuous-auth use cases.
- **Live mode busy-loops in the Streamlit thread**. The `while
  run_camera:` loop is fine for a demo but blocks reruns and will
  spin the CPU at full tilt on the inference path. Production should
  move this off-thread.
- **No face is analysed if YOLO doesn't fire**. If YOLO confidence on
  the only face in frame is below 0.50 (small face, low light, heavy
  occlusion), the layer reports "no faces detected" and produces no
  verdict at all. The static-upload path surfaces this as a warning;
  the live path silently shows the raw frame.

---

## 4. Alternatives & complementary detectors

### 4.1 Hand-coded frequency features (the previous Layer 2 design)
Spectral slope + HF peak count + windowed FFT on a fixed ROI.
- **Pros:** No training data, fully explainable, near-zero compute.
- **Cons:** Slope alone misses many modern diffusion deepfakes; peak
  count needs visible Moiré, which modern phone ISPs suppress.

### 4.2 Specialised PAD CNNs (DeepPixBiS, CDCN, Auxiliary)
End-to-end face anti-spoofing models from the PAD literature.
- **Pros:** SOTA accuracy on CelebA-Spoof, OULU-NPU benchmarks.
- **Cons:** Heavy (10–100 M params), domain-shift sensitive, opaque.

### 4.3 Vision–language detectors (CLIP-based)
Use CLIP embeddings as features for a small spoof classifier head.
- **Pros:** Generalises across attack types, single model.
- **Cons:** ~100 ms inference, opaque, adversarially fooled by
  text-prompted attacks.

### 4.4 rPPG (remote photoplethysmography)
Recover heart-rate signal from subtle skin colour pulsation across
~10 s of video.
- **Pros:** Proves *liveness* — a printed photo has no pulse signal,
  period.
- **Cons:** Needs stable video, good lighting, ~10 s capture window;
  sensitive to motion and skin-tone bias.

### 4.5 Active liveness challenges
Ask the user to blink, smile, or turn their head.
- **Pros:** Different attack surface (must respond to a specific
  prompt); catches replay attacks.
- **Cons:** Worse UX; defeated by real-time deepfake puppeteering or
  pre-recorded responses to known prompts.

### 4.6 Depth / 3D liveness (FaceID-style)
Use a structured-light or ToF sensor to get a depth map and check
whether the "face" has actual 3D structure.
- **Pros:** Cryptographically strong against any 2D attack.
- **Cons:** Hardware-locked to specific phones / dedicated rigs.

---

## 5. How this layer fits into the full stack

| Layer | Catches                                  | Cost       |
| ----- | ---------------------------------------- | ---------- |
| 1     | Injection (virtual cams, file replay)    | µs/frame   |
| **2** | **Deepfake / screen / print (YOLO + FFT)** | **~30 ms/frame** |
| 3     | Print / screen physical texture (LBP)    | ms/frame   |
| 4     | Identity match (InsightFace embeddings)  | ~100 ms    |

Layer 2 is the "brains" of the visual gate — it makes the per-frame
real/fake call. Layer 3 then adds an independent texture-based
sanity check, and Layer 4 confirms identity once a frame passes both
gates.

---

## 6. Dependencies

- `ultralytics` — YOLO inference.
- `cvzone` — `cornerRect` and `putTextRect` overlay helpers.
- `opencv-python` — DFT, color conversion, video capture.
- `Pillow` + `numpy` + `streamlit` — image I/O and UI.
- **A trained `best.pt`** placed alongside `layer2_deepfake.py`.

If `best.pt` is missing, `load_yolo_model()` raises and the UI shows
the failure message instead of running.

---

## 7. Production recommendations

1. **Ship the model with the binary** (or fetch it from a signed URL
   on first launch) so users don't see "Failed to load YOLO model"
   on a fresh install.
2. **Validate `CLASS_NAMES` on first run** — log a few predictions on
   known-real and known-spoof samples to confirm the index order
   matches your training set.
3. **Calibrate `FFT_VARIANCE_THRESHOLD` per device class** — collect
   ~100 real captures per browser/OS/sensor combo and pick the 99th
   percentile of the legitimate distribution as the threshold.
4. **Move live inference off the Streamlit script thread** — for
   production use `streamlit-webrtc` or run YOLO in a worker process.
   The current `while run_camera:` loop blocks reruns and burns CPU.
5. **Log both individual votes** (YOLO label, YOLO conf, FFT
   variance) alongside the final ensemble decision. False positives /
   negatives are far easier to triage when you can see which side of
   the ensemble fired.
6. **Combine with Layer 3 (LBP texture) and Layer 4 (identity)** —
   one detector is never enough for a high-stakes biometric gate.
