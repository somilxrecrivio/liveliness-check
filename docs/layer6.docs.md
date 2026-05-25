# Layer 6 · Video Liveness (rPPG + Micro-movement)

> **File:** [`layer6_video_liveness.py`](../layer6_video_liveness.py)
> **Goal:** Detect spoofs that pass every single-frame check —
> high-resolution prints, screen replays, deepfake stills — by
> measuring two signals that **only exist across multiple frames of a
> truly living subject**: heart-rate pulse and irregular micro-movement.

---

## 1. What it does

This is the first **multi-frame layer**. It opens the webcam, captures
~10 s of video at ~30 fps, runs per-frame face detection + cheek-ROI
mean RGB extraction, and computes three flags from the resulting
time-series.

```
[start capture]
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  cv2.VideoCapture loop @ ~30 fps for ~10 s              │
│    per-frame:                                            │
│      MediaPipe BlazeFace  →  bbox + 6 keypoints          │
│      extract cheek patch (eye↔mouth midpoint, edge-shifted)│
│      store: timestamp, bbox, keypoints, cheek_RGB_mean    │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────┐     ┌───────────────────────────┐
│ rPPG (CHROM algorithm)     │     │ Micro-movement analyser   │
│  • normalise RGB           │     │  • track nose+mouth midpt │
│  • X = 3R−2G, Y = 1.5R+G−1.5B │     │  • Δ_t = ||p_t − p_{t−1}||│
│  • bandpass 0.7–4 Hz       │     │  • var(Δ), ACF(Δ)         │
│  • pulse = X−αY            │     │                           │
│  • FFT → SNR, peak freq    │     │  • motion_variance        │
└────────────────────────────┘     │  • periodicity = max|ACF| │
       │                           │    for lag > 0.5 s        │
       ▼                           └───────────────────────────┘
   rPPG.snr                                     │
       │                            motion.motion_variance
       │                            motion.periodicity
       ▼                                        │
SNR < threshold → FLAG          var < min → FLAG │
                              periodicity > max → FLAG
       │                                        │
       └──────────────────  OR  ────────────────┘
                            ▼
                       SPOOF or REAL
```

### 1.1 rPPG via the CHROM algorithm

Cardiovascular pulse causes minute (≈ 0.5 %) cyclic changes in the
optical absorbance of skin at the haemoglobin wavelengths — strongest
in the green band, weaker in red, weakest in blue. With ~10 s of
~30 fps video of a cheek patch we can recover the heart-rate signal
in the 0.7–4 Hz band.

We use **CHROM** (de Haan & Jeanne 2013), which is robust to
illumination changes:

```
R, G, B = cheek_mean_per_frame / cheek_mean_overall  − 1   # zero-centred
X = 3·R − 2·G
Y = 1.5·R + G − 1.5·B
bandpass X, Y to [0.7, 4.0] Hz
α = std(X_bp) / std(Y_bp)
pulse = X_bp − α·Y_bp
```

The X and Y projections are chosen to maximise the pulse signal while
suppressing the dominant skin-tone direction (so the method works
across ethnicities without re-calibration).

**SNR** = peak-band-power / median-band-power in the FFT of the
pulse. A real cheek produces a strong, narrow peak in the
heart-rate band → high SNR. A printed photo or static screen has no
pulsatile variation → SNR ≈ 0.

### 1.2 Micro-movement

A live subject's head moves involuntarily even when they're "sitting
still": breathing translates the face vertically by tenths of a
pixel; microsaccades shift the eyes a few times per second; skin
micro-flexing perturbs the keypoints in tiny irregular ways. A
printed photograph on a stand has zero displacement. A looped video
replay has *periodic* displacement (the same head turn replaying every
few seconds).

We track the midpoint of the BlazeFace `nose_tip` and `mouth_center`
keypoints frame-to-frame:

```
positions = [(nose + mouth)/2 per frame]
displacements = ||positions[t+1] − positions[t]||
motion_variance = var(displacements)
acf = autocorrelation(displacements − mean) / acf[0]
periodicity = max(|acf|) for lag ∈ [0.5 s, 4 s]
```

- **Static photo:** `displacements ≈ 0` → `motion_variance ≈ 0` → FLAG.
- **Real face:** irregular jitter → moderate variance, low periodicity.
- **Looped replay:** periodic displacement → high `periodicity` → FLAG.

### 1.3 Storage strategy

The capture loop stores **per-frame features only** (timestamp,
bbox, 6 keypoints, 3 floats of cheek-RGB-mean), not the full
frames. Memory stays bounded at ~30 KB for 300 frames, which is
crucial for Streamlit's session-state caching to survive reruns.

---

## 2. Pros

- **The only true liveness signals in the stack.** Pulse and
  involuntary micro-movement do not exist in any 2-D or 3-D
  presentation attack. A perfectly-printed face on a real human
  hand still doesn't have a pulse on the *paper*.
- **Catches deepfakes the other layers cannot.** A photorealistic
  deepfake video at any resolution still doesn't render
  pulse-correlated skin-colour changes correctly — the model
  doesn't know the subject's heart rate.
- **Defeats simple replay attacks** via the periodicity flag —
  looped clips repeat motion patterns that real subjects don't.
- **No new model downloads beyond BlazeFace** (~230 KB) — uses
  the same face detector as Layers 3 and 5.
- **Independent failure modes from Layers 1–5.** Even an attacker
  who somehow passes texture, depth, FFT, and YOLO checks can't
  fake heart-rate phase coherence on the cheek.
- **Heart-rate estimate as a side benefit** — the same pipeline
  gives you a (rough) bpm number that can be displayed to the
  operator.

## 3. Cons & failure modes

- **Slow.** ~10 s of video is the bare minimum for stable
  pulse-FFT estimation; that's an order of magnitude longer than
  any single-frame layer. Bad UX for high-throughput onboarding.
- **Per-frame face detection load.** BlazeFace at ~10–30 ms/frame
  for 300 frames is 3–9 s of compute that runs *inside* the
  capture loop — risks dropping the effective fps.
- **rPPG SNR is fragile under:**
  - Very still subjects (the irony — too still is also bad).
  - Strong directional light variation across the capture
    window (cloud passing the window, fluorescent flicker).
  - Camera AGC / AWB updates during capture (modern smartphones
    aggressively rebalance white point — destroys pulse signal).
  - Low-light captures (sensor noise dwarfs the 0.5 % pulse
    modulation).
- **Periodicity false-positives.** Heartbeat itself is periodic.
  We restrict the periodicity flag to motion lag > 0.5 s
  (heart-rate periods are < 1 s, but a pulse-driven head bob is
  at the same period — could trigger). Calibrate the threshold
  empirically.
- **Periodicity false-negatives.** A clever attacker can play a
  long, varied loop that doesn't repeat within the capture
  window. Capture duration needs to be ≥ the suspected loop
  length.
- **Motion variance false-positives.** Subjects who don't sit
  still (anxious, restless, on a moving platform) produce so much
  motion that the displacement series exits the noise regime,
  and ACF analysis becomes unreliable.
- **Threshold defaults are starting points only.** Calibrate
  against labelled real-and-spoof captures from your target
  hardware and lighting conditions.
- **Streamlit doesn't support background capture cleanly.** The
  current implementation blocks the script thread during
  capture. Production deployments should move the loop into a
  worker thread or `streamlit-webrtc`.
- **No protection against deepfake video with rPPG injection.**
  Recent research (e.g. "DeepRhythm") shows it's possible to
  *render* a fake pulse into a deepfake's cheek pixels. The
  micro-movement and periodicity flags remain reliable, but the
  pulse flag alone is not future-proof.

---

## 4. Alternatives & complementary signals

### 4.1 POS (Plane-Orthogonal-to-Skin) rPPG
Wang et al. 2017's improvement on CHROM, using a different
orthogonal projection.
- **Pros:** Slightly better SNR on dark skin tones.
- **Cons:** Same overall failure modes; marginal improvement.

### 4.2 Deep rPPG (DeepPhys, PhysNet, TS-CAN)
End-to-end neural pulse extractors.
- **Pros:** SOTA SNR on benchmark datasets.
- **Cons:** Need GPU for real-time; opaque; adversarially
  attackable; can be fooled by rPPG-injected deepfakes.

### 4.3 Active liveness challenges
Ask the user to blink, smile, turn their head, follow a moving
dot.
- **Pros:** Cheap, fast, robust against static attacks.
- **Cons:** UX friction; defeated by real-time deepfake
  puppeteering; replay attacks on known prompts.

### 4.4 Eye-gaze tracking
Verify the subject's gaze responds to on-screen prompts.
- **Pros:** Strong against video replay.
- **Cons:** Requires fine-grained gaze estimation; user must
  cooperate.

### 4.5 Blink detection
Count voluntary or involuntary blinks during the capture.
- **Pros:** Trivial signal (closed-eye frames are easy to
  detect); printed photos never blink.
- **Cons:** Pre-recorded video easily passes; deepfake
  generators learn to render blinks.

### 4.6 Audio-visual lip sync
Ask the subject to read a random sentence; verify lip motion
matches the audio.
- **Pros:** Defeats most pre-recorded attacks; audio adds an
  independent modality.
- **Cons:** UX friction; defeated by real-time deepfake
  + voice clone.

### 4.7 Hardware: ToF / structured-light depth + RGB
Sensor-level liveness (FaceID).
- **Pros:** Strongest single signal available.
- **Cons:** Hardware-locked; not portable to web/laptop flows.

---

## 5. How this layer fits into the full stack

| Layer | Catches                                  | Cost            | Frames |
| ----- | ---------------------------------------- | --------------- | ------ |
| 1     | Injection (virtual cams, file replay)    | µs/frame        | 1+     |
| 2     | Deepfake / screen / print (YOLO + FFT)   | ~30 ms/frame    | 1      |
| 3     | Texture (LBP + Laplacian + Moiré + IQA)  | ~10 ms/frame    | 1      |
| 4     | Identity match (InsightFace ArcFace)     | ~100 ms/face    | 1      |
| 5     | 2-D vs 3-D depth (Depth Anything)        | ~200 ms/face    | 1      |
| **6** | **Pulse + micro-movement (CHROM + ACF)** | **~10–15 s capture + ~1 s analysis** | **~300** |

Layer 6 is by far the slowest, but it's the **only layer that
verifies the subject is biologically alive** in this capture, right
now. The other layers verify that the *image content* is consistent
with a live face; Layer 6 verifies that *the act of capturing*
recorded a pulse and natural micro-motion.

The recommended order:

1. **Layers 1–4 as fast filters.** Reject obvious spoofs in
   milliseconds.
2. **Layer 5 for ambiguous mid-quality spoofs** (when 1–4 disagree).
3. **Layer 6 for the gold standard** — used at enrolment, periodic
   re-verification, or any time the upstream stack returns
   borderline scores. Not for the every-request fast path.

---

## 6. Testing recipes

1. **Sit ~50 cm from the webcam, even lighting, look at the screen
   for 10 s** → expect REAL. SNR > 5, heart rate 60–90 bpm,
   motion variance 0.5–3 px², periodicity < 0.4.

2. **Print a photo of yourself, hold to the webcam, stay still for
   10 s** → expect SPOOF. Both pulse and motion flags fire.

3. **Show your face on your phone, hold to the webcam** → expect
   SPOOF. Pulse flag fires (the phone doesn't render your live
   pulse).

4. **Loop a recorded 5-second video of yourself, capture for
   10 s** → expect SPOOF (periodicity flag fires because motion
   repeats with period 5 s).

5. **Real face, you deliberately move your head a lot** → may
   false-positive on periodicity if the movement is rhythmic
   (e.g. nodding). Real verification should ask the user to sit
   still.

6. **Real face but with strong lighting changes** (window with
   passing clouds) → pulse SNR drops, may false-positive. The
   subject should be in stable lighting for the capture window.

If a real subject fails the pulse flag, look at:
- The **pulse FFT chart** — is there ANY peak in the 0.7–4 Hz
  band? If not, the cheek RGB is too quiet (low light, too dark
  skin, AGC instability).
- The **heart-rate estimate** — is it physiologically plausible
  (40–200 bpm) or random?

If a real subject fails the periodicity flag, look at:
- The **motion ACF chart** — is there a clear repeating peak?
  If yes, the subject is doing something rhythmic (breathing
  heavily, tapping foot transferring to head).

---

## 7. Recommendations for production

1. **Move the capture loop off the Streamlit thread.** Use
   `streamlit-webrtc`, a worker process, or a JS frontend that
   uploads the captured frames in one batch. The current
   blocking loop kills concurrency.
2. **Calibrate thresholds per skin-tone bucket.** rPPG SNR varies
   measurably with melanin content; using a single threshold
   systematically under-credits darker-skinned subjects (a
   well-documented bias in published rPPG benchmarks).
3. **Capture in stable lighting.** Add a pre-flight check (Layer
   3's IQA quality gate) that aborts before the 10 s capture if
   lighting is bad — saves the user a wasted attempt.
4. **Combine with Layer 5 (depth).** Depth catches static 2-D
   attacks even when the user is asked to move; rPPG catches
   high-quality replays even when geometry looks right. Together
   they cover near-orthogonal attack surfaces.
5. **Log the captured features for forensics.** Cheek-RGB time
   series, keypoint trajectories, and the resulting pulse / ACF
   are tiny (~30 KB) — keep them per claim so reviewers can
   triage borderline verdicts.
6. **Don't rely on rPPG alone against advanced deepfakes.** The
   periodicity + motion-variance flags are more robust against
   rPPG-injected fakes; weight them at least as heavily in the
   ensemble.
7. **Tune capture duration to the use case.** 10 s is the
   minimum for stable FFT; KYC enrolment can afford 15–20 s
   for better SNR; continuous-auth flows might need to drop to
   5 s + lower SNR threshold + ensemble vote across multiple
   short captures.
