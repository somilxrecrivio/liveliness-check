# Layer 5 · Monocular Depth PAD

> **File:** [`layer5_depth_pad.py`](../layer5_depth_pad.py)
> **Goal:** Detect 2-D presentation attacks (printed photos, displayed
> screens) by checking whether the face crop has the depth structure
> of a real 3-D face. Powered by **Depth Anything V2 Small** (ONNX,
> CPU).

---

## 1. What it does

A real face is a 3-D object — the nose protrudes 2–4 cm beyond the
cheeks, which are themselves closer than the ears / background.
A printed photo or a screen replay is flat — the depicted face sits
on a 2-D surface, so every face-pixel has the same physical depth as
every other face-pixel.

Pipeline:

```
image  →  EXIF  →  resize ≤1024  →  MediaPipe BlazeFace  →  face bbox
                                                              │
                                                              ▼
                                                  crop with 15% margin
                                                  (corners include cheek/background)
                                                              │
                                                              ▼
                                          Depth Anything V2 Small (518×518 ONNX)
                                                              │
                                                              ▼
                                              inverse-depth map (raw)
                                                              │
                                                              ▼
                                              normalise to [0, 1]
                                                              │
                              ┌────────────────────────────────┴──────────────────────┐
                              ▼                                                       ▼
                  variance of normalised map                    mean(central 30%) − mean(corner 15% × 4)
                              │                                                       │
                              ▼                                                       ▼
                         < 0.03  → FLAG                                            < 0.10  → FLAG
                              │                                                       │
                              └──────────────────────  OR  ──────────────────────────┘
                                                              ▼
                                                       SPOOF or REAL
```

### Why both flags

A 2-D photo can sometimes pass *one* of the two checks alone:

- **Flat homogeneous photo** → fails variance, passes centre−edge (no centre, no edge, depth ≈ uniform).
- **Photo of a photo at an angle** → may have non-trivial variance from perspective rendering, but the centre-vs-edge gradient is still wrong.

Running both flags catches both modes.

### Why crop with margin

The corners of the face crop need to contain something *that isn't
the face* (cheek edge, hair, background wall) so the depth model can
report a depth gradient between the protruding nose and the receding
corners. A tight crop with the face filling 100 % of the frame
collapses the centre-vs-edge signal even on real subjects. The
default 15 % margin around the BlazeFace bbox solves this.

### Why Depth Anything V2 Small

- **State of the art for monocular depth** at this size class
  (~99 MB ONNX, ~25 M params). Trained on a large mix of synthetic
  + real-world data; generalises far better than older MiDaS small.
- **CPU-friendly** — ~150–400 ms per crop on modern CPUs.
- **ONNX Runtime** with the existing `onnxruntime` install (already
  on disk for Layer 4). No new heavy dependencies.
- **Stable HuggingFace mirror** at `onnx-community/depth-anything-v2-small`.
  Auto-downloads on first run to `./models/`.

---

## 2. Pros

- **Attacks a class of spoof signals upstream layers don't.** LBP
  (Layer 3) measures texture; this measures geometry. A
  high-quality print with realistic micro-texture (good paper,
  high-DPI inkjet) might pass LBP but cannot fake 3-D depth.
- **Single-frame.** No video required, no multi-frame
  bookkeeping.
- **No labelled training data needed** — the depth model is
  pretrained on general-purpose imagery; we only threshold its
  outputs.
- **Visual evidence.** The VIRIDIS-coloured depth map *shows* the
  reviewer whether the face has nose-cheek-corner structure or is
  uniformly coloured (flat).
- **Independent failure mode from Layers 2 and 3.** Even a perfectly
  texture-matched, perfectly Moiré-free deepfake screen replay will
  appear flat to the depth network.

## 3. Cons & failure modes

- **Sensitive to crop framing.** If the BlazeFace bbox is too tight
  (face fills the crop, no margin), centre-vs-edge collapses to
  zero on real subjects too. The 15 % margin helps but is not
  bulletproof — extreme close-ups or sideways faces can produce
  false positives.
- **The depth model can hallucinate structure on photos with
  shading.** A high-resolution print under directional light still
  has self-shadows that the network interprets as depth cues. The
  centre-vs-edge gap shrinks but doesn't always reach zero.
- **Sensitive to camera distance.** Very close captures (the face
  fills the frame) compress the depth gradient even on real
  subjects.
- **The model is 99 MB.** First-run download is a noticeable wait.
  Pre-stage in production.
- **Slower than the rest of the stack.** ~150–400 ms vs Layer 3's
  ~10 ms. Reserve for cases that need disambiguation, not the
  every-frame fast path.
- **Threshold calibration is empirical.** Variance and
  centre-vs-edge ranges depend on camera FoV, subject distance,
  and lighting. Calibrate per device class.
- **Cannot tell a photograph of a 3-D mannequin from a real face.**
  Sculpted attacks (silicone masks, 3-D prints) defeat this layer.
  Used alone, it's not a complete liveness check.

---

## 4. Alternatives & complementary signals

### 4.1 MiDaS Small / DPT Hybrid
Older / larger monocular depth estimators.
- **Pros:** Mature, well-documented; DPT Hybrid is more accurate.
- **Cons:** MiDaS Small is noisier than Depth Anything V2 Small;
  DPT Hybrid is too heavy (~300 MB).

### 4.2 ZoeDepth
Metric depth instead of relative depth.
- **Pros:** Outputs physical-scale depth, which gives more
  interpretable thresholds.
- **Cons:** Larger model; metric calibration is sensitive to
  scene content for the kind of close-up face crops we feed it.

### 4.3 Structured-light / ToF sensors (FaceID, OAK-D, RealSense)
Hardware depth sensors.
- **Pros:** Cryptographically strong against any 2-D attack;
  far more accurate than monocular estimation.
- **Cons:** Hardware-locked to specific devices; not portable to
  generic webcam/browser flows.

### 4.4 Stereo depth (two cameras)
- **Pros:** Cheap, deployable on dual-camera phones.
- **Cons:** Requires aligned hardware; deployment-coupled.

### 4.5 Photometric-stereo / shape-from-shading
Single-camera depth via lighting cues.
- **Pros:** No model needed; pure linear algebra.
- **Cons:** Needs known illumination; fragile in casual selfie
  capture conditions.

### 4.6 3-D face landmarks (MediaPipe FaceLandmarker)
Returns 468 landmarks in 3-D — but the depth coordinates are
*canonical face shape*, not scene depth.
- **Pros:** Free with MediaPipe.
- **Cons:** Does **not** discriminate 2-D from 3-D — the model
  outputs canonical face depth even for a printed photo of a face.

### 4.7 Sculpted-attack detection (anti-mask)
Multi-spectral imaging (near-IR + visible), or shape consistency
checks against the MediaPipe canonical mesh.
- **Pros:** Catches silicone / 3-D-printed masks that defeat
  depth-only PAD.
- **Cons:** Needs hardware (NIR) or a more involved model.

---

## 5. How this layer fits into the full stack

| Layer | Catches                                  | Cost            |
| ----- | ---------------------------------------- | --------------- |
| 1     | Injection (virtual cams, file replay)    | µs/frame        |
| 2     | Deepfake / screen / print (YOLO + FFT)   | ~30 ms/frame    |
| 3     | Print / screen texture (LBP + Laplacian + Moiré + IQA) | ~10 ms/frame |
| 4     | Identity match (InsightFace embeddings)  | ~100 ms/face    |
| **5** | **2-D vs 3-D structure (depth model)**   | **~150–400 ms/face** |

Layer 5 is the **slowest** layer and attacks a class of signals
the others miss. Run it when the cheap layers disagree, or on every
high-stakes verification.

Defence-in-depth ordering with Layer 5 added:

1. Layer 1: real hardware?
2. Layer 2: deepfake or screen replay?
3. Layer 3: live skin micro-texture?
4. **Layer 5: actual 3-D structure?**
5. Layer 4: matches the reference identity?

---

## 6. Testing recipes

1. **Webcam → your face, ~50 cm from camera** → expect REAL. Variance
   ≥ 0.05, centre−edge ≥ 0.15.

2. **Print a photo of yourself on plain paper, hold to webcam** →
   expect SPOOF. Variance often still passes (paper has shading
   cues) but centre−edge typically drops below 0.10 because the
   "nose" pixels are at the same physical depth as the "ear"
   pixels (both on the paper surface).

3. **Show a face on your phone screen, photograph with webcam** →
   expect SPOOF. Same physics as the print case — screen surface
   is flat.

4. **Face filling the entire frame (very close)** → may
   false-positive on real subjects because there's no background
   in the corners. Step back from the camera.

5. **Sideways profile** → BlazeFace may miss, falling through with
   no verdict. Use a frontal pose.

If a real subject is misclassified, look at the depth map:
- If the face is uniformly coloured → the model didn't pick up
  depth cues. Try with more background visible (step back).
- If the corners are brighter than the centre → the framing
  has the face touching the bbox edges. Add margin / step back.
- If the depth map is dim everywhere → too dark a capture.

---

## 7. Recommendations for production

1. **Pre-stage the 99 MB depth model** in the container image.
   First-run downloads kill UX.
2. **Calibrate thresholds per camera FoV / distance class.** Phone
   selfies, laptop webcams, and DSLR portraits all produce different
   centre-vs-edge ranges for the same scene.
3. **Combine with Layer 3.** Layer 3 alone passes most high-quality
   prints; Layer 5 alone passes silicone masks. Together they cover
   both attack modes.
4. **Reserve Layer 5 for risk-weighted use** — it's the slowest
   layer in the stack. Run on enrolment and on flagged
   verifications; skip on the cheap fast path if Layers 1–3 all
   pass cleanly.
5. **Log the depth map.** A reviewer triaging a false positive can
   immediately tell from the depth image whether the model
   misjudged scene geometry.
6. **Stack with rPPG (Layer 6)** for the strongest single-vendor
   passive-only PAD coverage we can build without specialised
   hardware.
