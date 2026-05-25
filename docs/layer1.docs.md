# Layer 1 · Hardware & Metadata IAD

> **File:** [`layer1_metadata.py`](../layer1_metadata.py)
> **Goal:** Detect *injected* video streams (virtual cameras, OBS, deepfake bots piping pre-rendered files) **before** any pixel-level analysis runs.

---

## 1. What it does

The layer treats the **inter-frame timestamp series** as a signal and asks one
question: *does it look like it came from real hardware, or from software that
synthesised the timing?*

For a sequence of frame timestamps `t₀, t₁, … tₙ` we compute:

```
δᵢ      = tᵢ₊₁ − tᵢ              # inter-frame delta
σ²(δ)   = Var({δᵢ})              # the headline metric
```

| Source                              | Expected δ          | Expected σ²(δ) |
| ----------------------------------- | ------------------- | --------------- |
| Real webcam @ 30 fps                | ≈ 33.3 ms ± 1–5 ms  | ~1 – 25 ms²     |
| Virtual cam replaying a video file  | ≈ 33.3 ms exactly   | ≈ 0 ms²        |
| File reader / deepfake pipeline     | Often clamped to fps | ≈ 0 ms²        |

If `σ²(δ) < threshold` the stream is flagged as **INJECTION SUSPECTED**.

The prototype offers **three frame sources** so you can validate the logic
without hardware *and* test against a real camera:

| Source                          | Function                       | Purpose                                                                 |
| ------------------------------- | ------------------------------ | ----------------------------------------------------------------------- |
| Bot (simulated injection)       | `generate_bot_timestamps`      | Synthetic perfect 33.33 ms cadence — sanity-checks the detector math.   |
| Real camera (simulated jitter)  | `generate_real_timestamps`     | Synthetic Gaussian jitter — lets you tune the threshold without hardware. |
| **Live webcam capture**         | `capture_live_timestamps`      | Opens the real camera via `cv2.VideoCapture` and records per-frame arrival times. |

### Testing against a real injection

To validate the **bot side** end-to-end:

1. Install **OBS Studio** and enable the **OBS Virtual Camera**.
2. Add a Media Source pointing at any recorded MP4.
3. Start the virtual camera, then in the prototype set **Camera index** to
   1 (or whichever index the virtual cam landed on — check
   `cv2.VideoCapture(i).isOpened()` if unsure) and click **Capture**.
4. The variance should collapse toward 0 ms², and the verdict should flip
   to **INJECTION SUSPECTED** even though OpenCV is reading "real" frames.

For the **real side**, just point the prototype at your built-in webcam
(`Camera index = 0`).

### What we actually measure in live capture

`capture_live_timestamps` records `time.perf_counter_ns()` in userspace
*after* `cv2.VideoCapture.read()` returns. That means the variance we
measure is the **sum of**:

- sensor + ISP timing variance,
- USB/MIPI transport delay,
- OS scheduling jitter on our own process,
- `cv2` decode time.

Items 3 and 4 add *more* jitter than a true hardware timestamp would, so
the live signal is even easier to distinguish from an injected one in this
prototype. In production you'd switch to a hardware-level timestamp
(V4L2 buffer ts, AVFoundation `CMSampleBufferGetPresentationTimeStamp`,
WebRTC `captureTimeMs`) to tighten the legitimate distribution and make
threshold tuning more reliable.

### Why this works physically

A real capture pipeline accumulates jitter from many independent sources:

- OS scheduler — the userspace callback isn't woken at a perfectly periodic
  rate
- Sensor readout — exposure time + rolling shutter introduce variance
- USB / MIPI bus contention — frames queue behind other traffic
- ISP processing — auto-exposure, white balance, demosaic timing is content-dependent

The Central Limit Theorem turns the sum of those independent noise sources
into an approximately Gaussian jitter distribution. Software that re-emits a
file at a constant FPS gets none of that for free.

---

## 2. Pros of the variance approach

- **Zero ML, zero training data.** Pure deterministic math: `np.diff` →
  `np.var`. Runs in microseconds.
- **Cheap to deploy.** No model artifacts, no GPU, no licensing cost.
- **Explainable.** A single number with a clear physical interpretation —
  easy to audit and tune for regulators / risk teams.
- **Modality-agnostic.** Works for any stream that exposes per-frame
  timestamps (WebRTC `RTCRtpReceiver.getStats()`, MediaRecorder, OpenCV
  `CAP_PROP_POS_MSEC`, mobile camera2 buffer timestamps).
- **Catches the entire "lazy attacker" class.** Anyone running OBS Virtual
  Camera, ManyCam, or piping ffmpeg through `v4l2loopback` fails this
  test immediately.
- **Composable.** Runs in parallel with every other layer at negligible cost;
  perfect as a cheap front-line filter before invoking expensive deepfake
  models.

## 3. Cons & failure modes

- **Sophisticated attackers can fake jitter.** An adversary aware of this
  check can sample noise from `N(33.3, σ²)` and resample frames to that
  cadence — the math becomes indistinguishable from a real camera. This is
  a **necessary, not sufficient** signal.
- **Requires trustworthy timestamps.** Wall-clock (`time.time()`) timestamps
  applied *after* receiving the frame measure your own scheduler, not the
  camera's. You want hardware-level timestamps (Android `Image.timestamp`,
  iOS `CMSampleBufferGetPresentationTimeStamp`, V4L2 buffer timestamp). On
  the web, WebRTC `captureTimeMs` is the right hook.
- **Some virtual cameras DO add jitter.** Modern OBS forks and screen-share
  pipelines re-clock frames through a real scheduling layer that introduces
  incidental jitter, leading to false negatives.
- **Burst / dropped frames look like jitter.** A laggy laptop can produce a
  high variance even on injected content if the playback is uneven —
  false positive risk.
- **Threshold is empirical and platform-dependent.** A high-end DSLR
  produces less jitter than a $5 webcam; a phone in low light produces
  more jitter than the same phone in good light (exposure time varies).
  You need to calibrate per device class.
- **Requires enough frames.** Variance estimates are unstable below ~30
  samples — you need ~1 s of video before the decision is reliable.
- **Says nothing about presentation attacks.** A printed photo waved in
  front of a real webcam passes this test perfectly.

---

## 4. Alternatives & complementary signals

### 4.1 PRNU (Photo-Response Non-Uniformity) sensor fingerprinting

Each CMOS sensor has a unique multiplicative noise pattern caused by
silicon manufacturing variance. Extracting and matching it across frames
identifies the physical sensor.

- **Pros:** Hardware-level uniqueness; effectively impossible to spoof
  without physical access to the original device.
- **Cons:** Requires per-user enrollment, fragile across firmware updates,
  computationally heavy (denoising filter per frame), defeated by lossy
  re-encoding.

### 4.2 Hardware attestation (TEE / Secure Enclave)

Android Key Attestation, iOS DeviceCheck/App Attest, or Play Integrity API
let the OS cryptographically assert "this frame came through the real
camera HAL on a non-rooted device."

- **Pros:** Strongest possible signal; cryptographically rooted in
  trusted hardware.
- **Cons:** Platform-specific (no equivalent on desktop web); breaks for
  legitimate users on rooted/jailbroken devices; rolls out slowly.

### 4.3 OS-level device enumeration

Inspect the camera device descriptor (`Vendor ID`, `Product ID`, driver
name, AVCaptureDevice attributes) and reject known virtual-cam drivers.

- **Pros:** Trivial to implement; catches the big public virtual cams by
  name.
- **Cons:** Easily bypassed — most virtual cam drivers can rename
  themselves; new ones appear faster than blacklists update.

### 4.4 Higher-order statistical tests on the timing series

Instead of variance, test the *distribution shape* of δ:

- **Allan deviation / power spectral density** — real cameras exhibit
  characteristic 1/f noise; bots emit white noise or a delta function.
- **Autocorrelation** — real jitter has short correlation lengths from
  shared scheduling resources; synthetic jitter is i.i.d.
- **Kolmogorov–Smirnov / Anderson–Darling** against a learned per-device
  template.

- **Pros:** Catches naive jitter injection ("attacker adds Gaussian
  noise") because the *shape* of the noise is wrong.
- **Cons:** Needs more frames (~5–10 s), more compute, and a calibration
  step per device class to learn the legitimate distribution.

### 4.5 Active challenge–response (active liveness)

Ask the user to blink, turn their head, or follow a moving dot, and
verify the response timing.

- **Pros:** Probes a different attack surface (the attacker must respond
  *to a specific prompt*, not just play a video).
- **Cons:** Worse UX, slower onboarding, defeated by pre-recorded
  responses to known prompts or by real-time deepfake puppeteering.

### 4.6 Audio-visual synchrony

If audio is available, correlate lip movement (visual) with phoneme
onsets (audio). Out-of-sync = injected.

- **Pros:** Very hard to fake live; complements pixel-domain liveliness.
- **Cons:** Requires audio capture, fails for silent flows (KYC selfies),
  and modern lip-sync deepfake models close the gap.

### 4.7 Network-layer fingerprinting

User-agent, TLS JA3, WebRTC ICE candidates, and TURN-vs-host signals
distinguish browser-based attackers from native captures.

- **Pros:** Free signal that runs at the edge.
- **Cons:** Trivially spoofed by anyone using a real browser; useless
  against native mobile-app attackers.

---

## 5. How this layer fits into the full stack

| Layer | Catches                                  | Cost     |
| ----- | ---------------------------------------- | -------- |
| **1** | Injection (virtual cams, file replay)    | µs/frame |
| 2     | Deepfake / screen Moiré (FFT)            | ms/frame |
| 3     | Print / screen presentation (LBP + blur) | ms/frame |
| 4     | Identity match (InsightFace embeddings)  | ~100 ms  |

Layer 1 is intentionally the **cheapest** check and runs first. It rejects
the majority of low-effort attacks before any expensive ML model is even
loaded.

---

## 6. Recommendations for production

1. **Use hardware timestamps**, not wall-clock. Tap into the camera HAL /
   WebRTC `captureTimeMs`.
2. **Calibrate per device class.** Collect baseline variance distributions
   from a few hundred sessions per browser/OS/sensor combination and set
   the threshold at the 1st percentile of the legitimate distribution.
3. **Use a higher-order test alongside variance** (autocorrelation or
   Allan deviation) to defeat the obvious "attacker adds Gaussian noise"
   bypass.
4. **Aggregate with later layers.** Treat the variance as one signal in a
   logistic-regression or boosted-tree risk score, not a hard gate.
5. **Log raw deltas, not just the verdict.** When false positives are
   reported you'll want the timing series for forensics.
