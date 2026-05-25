"""
Layer 6: Video Liveness (rPPG + Micro-movement)
================================================

Concept
-------
Single-frame PAD layers can be fooled by a high-quality static
spoof: a perfect-resolution print, a 4K screen replay at matched
focus. Two physical signals only exist in real, *living* faces and
only show up across multiple frames:

1. **Pulse signal (rPPG)** — minute green-band oxygenation changes
   in cheek skin cause sub-pixel-level cyclic RGB variation at the
   subject's heart rate (~0.7–4 Hz, i.e. 42–240 bpm). A printed
   photo / static screen has zero pulse signal, ever.

2. **Involuntary micro-movement** — a live subject's head jitters
   constantly from breathing, microsaccades, and skin micro-flexing.
   A static photo has zero displacement; a looped video replay has
   *periodic* displacement (the same head turn repeating every
   few seconds).

We capture ~10 s of video at ~30 fps, per-frame face detection +
cheek-ROI mean RGB extraction, then compute three flags:

- **rPPG SNR** (peak-band-power / median-band-power in the FFT of
  the CHROM pulse signal). Below threshold → no pulse → SPOOF.
- **Motion variance** of frame-to-frame head displacement. Below
  threshold → too still → static spoof.
- **Motion periodicity** (max absolute autocorrelation at lag > 0.5 s).
  Above threshold → periodic motion → looped replay.

ANY flag → SPOOF.

Run:
    streamlit run layer6_video_liveness.py
"""

import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BLAZEFACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
BLAZEFACE_MODEL_PATH = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")

DEFAULT_CAPTURE_SECONDS = 10.0
DEFAULT_TARGET_FPS = 30.0
RPPG_BAND_HZ = (0.7, 4.0)   # 42-240 bpm, the physiological heart-rate band

DEFAULT_RPPG_SNR_THRESHOLD = 3.0
DEFAULT_MOTION_VAR_MIN = 0.5
DEFAULT_PERIODICITY_MAX = 0.45
DEFAULT_MIN_FACE_CONFIDENCE = 0.4

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
KEYPOINT_NAMES = [
    "right_eye",
    "left_eye",
    "nose_tip",
    "mouth_center",
    "right_ear_tragion",
    "left_ear_tragion",
]


@dataclass
class FrameFeatures:
    """Per-frame extracted features (small enough to keep all of them)."""
    timestamp: float
    bbox: Optional[tuple]
    keypoints: Optional[dict]
    cheek_rgb_mean: Optional[tuple]


@dataclass
class CaptureResult:
    features: list
    actual_fps: float
    last_frame: np.ndarray
    n_face_frames: int


@dataclass
class RPPGAnalysis:
    pulse_signal: np.ndarray
    fft_freqs: np.ndarray
    fft_power: np.ndarray
    heart_rate_hz: float
    heart_rate_bpm: float
    snr: float


@dataclass
class MotionAnalysis:
    displacements: np.ndarray
    motion_variance: float
    acf: np.ndarray
    periodicity: float


@dataclass
class Verdict:
    spoof: bool
    reasons: list
    flag_pulse: bool
    flag_motion: bool
    flag_periodicity: bool


# ---------------------------------------------------------------------------
# Face detection helpers (same approach as Layers 3 and 5)
# ---------------------------------------------------------------------------

_FACE_DETECTOR = None
_FACE_DETECTOR_CONF: Optional[float] = None


def _ensure_download(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _get_face_detector(min_confidence: float):
    global _FACE_DETECTOR, _FACE_DETECTOR_CONF
    if _FACE_DETECTOR is None or _FACE_DETECTOR_CONF != min_confidence:
        path = _ensure_download(BLAZEFACE_MODEL_URL, BLAZEFACE_MODEL_PATH)
        opts = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=path),
            min_detection_confidence=min_confidence,
        )
        _FACE_DETECTOR = FaceDetector.create_from_options(opts)
        _FACE_DETECTOR_CONF = min_confidence
    return _FACE_DETECTOR


def detect_face_with_kps(frame_rgb: np.ndarray, min_confidence: float):
    """Returns (bbox, kps_dict) or None."""
    detector = _get_face_detector(min_confidence)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)
    if not result.detections:
        return None
    detections = sorted(
        result.detections,
        key=lambda d: d.bounding_box.width * d.bounding_box.height,
        reverse=True,
    )
    det = detections[0]
    h, w = frame_rgb.shape[:2]
    bb = det.bounding_box
    bbox = (
        max(0, int(bb.origin_x)),
        max(0, int(bb.origin_y)),
        max(1, int(bb.width)),
        max(1, int(bb.height)),
    )
    kps = {
        name: (float(kp.x) * w, float(kp.y) * h)
        for name, kp in zip(KEYPOINT_NAMES, det.keypoints)
    }
    return bbox, kps


def extract_cheek_rgb(frame_rgb: np.ndarray, bbox, kps) -> Optional[tuple]:
    """Mean RGB of a cheek patch between the right-eye and mouth keypoints."""
    eye = kps["right_eye"]
    mouth = kps["mouth_center"]
    cx, cy = (eye[0] + mouth[0]) / 2, (eye[1] + mouth[1]) / 2
    bx, by, bw, bh = bbox
    # Push toward the face edge to land squarely on cheek skin.
    if cx < bx + bw / 2:
        cx = (cx + bx) / 2
    else:
        cx = (cx + bx + bw) / 2
    half = max(8.0, min(bw, bh) / 8.0)
    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    x1 = int(min(frame_rgb.shape[1], cx + half))
    y1 = int(min(frame_rgb.shape[0], cy + half))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    patch = frame_rgb[y0:y1, x0:x1]
    return tuple(float(v) for v in patch.mean(axis=(0, 1)))


# ---------------------------------------------------------------------------
# Video capture loop
# ---------------------------------------------------------------------------

def capture_video(
    duration_sec: float,
    target_fps: float,
    min_face_conf: float,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    preview_cb: Optional[Callable[[np.ndarray], None]] = None,
) -> CaptureResult:
    """
    Open the default webcam, capture frames for `duration_sec`, and extract
    per-frame face features. Returns a CaptureResult with one entry per
    captured frame (some may have no face).

    We store the cheek RGB triple (3 floats) per frame, not the full pixels,
    so memory stays bounded at ~30 KB for 300 frames.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera. On macOS the first run triggers a "
            "permissions prompt — grant access and try again."
        )

    # Warm-up: drop a few frames so AE/AWB settle and timestamps stabilise.
    for _ in range(5):
        cap.read()

    features: list[FrameFeatures] = []
    last_frame_rgb = None
    n_with_face = 0
    start_t = time.perf_counter()

    try:
        while True:
            now = time.perf_counter() - start_t
            if now >= duration_sec:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            last_frame_rgb = frame_rgb

            detection = detect_face_with_kps(frame_rgb, min_face_conf)
            if detection is not None:
                bbox, kps = detection
                cheek_rgb = extract_cheek_rgb(frame_rgb, bbox, kps)
                features.append(FrameFeatures(now, bbox, kps, cheek_rgb))
                if cheek_rgb is not None:
                    n_with_face += 1
            else:
                features.append(FrameFeatures(now, None, None, None))

            if progress_cb is not None:
                progress_cb(
                    min(now / duration_sec, 1.0),
                    f"{len(features)} frames captured · "
                    f"{n_with_face} with face",
                )
            if preview_cb is not None and len(features) % 4 == 0:
                preview_cb(frame_rgb)
    finally:
        cap.release()

    total = time.perf_counter() - start_t
    actual_fps = len(features) / total if total > 0 else 0.0

    return CaptureResult(
        features=features,
        actual_fps=actual_fps,
        last_frame=last_frame_rgb
        if last_frame_rgb is not None
        else np.zeros((100, 100, 3), dtype=np.uint8),
        n_face_frames=n_with_face,
    )


# ---------------------------------------------------------------------------
# rPPG via the CHROM algorithm (de Haan & Jeanne 2013)
# ---------------------------------------------------------------------------

def chrom_pulse(rgb_signal: np.ndarray, fps: float):
    """
    CHROM-method pulse signal extraction.

    Returns
    -------
    pulse        : 1-D filtered pulse signal
    fft_freqs    : 1-D positive frequencies (Hz)
    fft_power    : 1-D power spectrum at those frequencies
    peak_freq    : Hz of the dominant peak inside the HR band
    snr          : peak band-power / median band-power
    """
    N = rgb_signal.shape[0]
    if N < int(fps * 3):
        return np.zeros(0), np.zeros(0), np.zeros(0), 0.0, 0.0

    # Per-channel normalisation: subtract and divide by mean → unit-mean
    # zero-centred series (CHROM step 1).
    mean = rgb_signal.mean(axis=0)
    if (mean <= 0).any():
        return np.zeros(N), np.zeros(0), np.zeros(0), 0.0, 0.0
    norm = rgb_signal / mean - 1.0

    R, G, B = norm[:, 0], norm[:, 1], norm[:, 2]

    # The two CHROM projections that maximise signal/skin-tone separation.
    X = 3.0 * R - 2.0 * G
    Y = 1.5 * R + G - 1.5 * B

    # Bandpass to the heart-rate range.
    nyq = fps / 2.0
    low = RPPG_BAND_HZ[0] / nyq
    high = min(RPPG_BAND_HZ[1] / nyq, 0.99)
    if low <= 0 or high <= low or high >= 1.0:
        return np.zeros(N), np.zeros(0), np.zeros(0), 0.0, 0.0
    try:
        b, a = butter(3, [low, high], btype="band")
        # Need enough samples for filtfilt's edge padding.
        padlen = 3 * max(len(a), len(b))
        if N <= padlen + 1:
            return np.zeros(N), np.zeros(0), np.zeros(0), 0.0, 0.0
        X_bp = filtfilt(b, a, X)
        Y_bp = filtfilt(b, a, Y)
    except Exception:
        return np.zeros(N), np.zeros(0), np.zeros(0), 0.0, 0.0

    # CHROM mixing coefficient.
    sy = float(np.std(Y_bp))
    alpha = float(np.std(X_bp)) / sy if sy > 1e-9 else 1.0
    pulse = X_bp - alpha * Y_bp

    # FFT for the SNR / heart-rate estimate.
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    power = np.abs(np.fft.rfft(pulse)) ** 2

    in_band = (freqs >= RPPG_BAND_HZ[0]) & (freqs <= RPPG_BAND_HZ[1])
    if not in_band.any():
        return pulse, freqs, power, 0.0, 0.0

    band_power = power[in_band]
    band_freqs = freqs[in_band]
    peak_idx = int(np.argmax(band_power))
    peak_power = float(band_power[peak_idx])
    median_band = float(np.median(band_power))
    snr = peak_power / median_band if median_band > 0 else 0.0

    return pulse, freqs, power, float(band_freqs[peak_idx]), snr


def analyse_rppg(features: list, fps: float) -> Optional[RPPGAnalysis]:
    """Build the cheek-RGB time series and run CHROM on it."""
    rgb = [f.cheek_rgb_mean for f in features if f.cheek_rgb_mean is not None]
    if len(rgb) < int(fps * 3):
        return None
    rgb_signal = np.asarray(rgb, dtype=np.float64)
    pulse, freqs, power, peak_freq, snr = chrom_pulse(rgb_signal, fps)
    if pulse.size == 0:
        return None
    return RPPGAnalysis(
        pulse_signal=pulse,
        fft_freqs=freqs,
        fft_power=power,
        heart_rate_hz=peak_freq,
        heart_rate_bpm=peak_freq * 60.0,
        snr=snr,
    )


# ---------------------------------------------------------------------------
# Micro-movement: frame-to-frame keypoint displacement + autocorrelation
# ---------------------------------------------------------------------------

def analyse_motion(features: list, fps: float) -> Optional[MotionAnalysis]:
    """
    Compute frame-to-frame displacement of the nose+mouth midpoint and
    its autocorrelation.

    Why nose+mouth midpoint: of the 6 BlazeFace keypoints, those two are
    the most stable (eyes have blink artefacts; ears are off-screen for
    many shots). Averaging the two damps single-frame jitter.
    """
    pos_list = []
    for f in features:
        if f.keypoints is None:
            continue
        n = f.keypoints["nose_tip"]
        m = f.keypoints["mouth_center"]
        pos_list.append(((n[0] + m[0]) / 2.0, (n[1] + m[1]) / 2.0))

    if len(pos_list) < int(fps * 3):
        return None

    positions = np.asarray(pos_list, dtype=np.float64)
    displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    motion_variance = float(np.var(displacements))

    # Periodicity via the normalised autocorrelation of (displacement − mean).
    d = displacements - displacements.mean()
    if d.std() < 1e-9:
        acf = np.zeros_like(d)
        periodicity = 0.0
    else:
        full = np.correlate(d, d, mode="full")
        full = full[len(d) - 1 :]
        acf = full / full[0]
        min_lag = max(2, int(fps * 0.5))
        max_lag = min(len(acf) - 1, int(fps * 4.0))
        periodicity = (
            float(np.max(np.abs(acf[min_lag:max_lag])))
            if max_lag > min_lag
            else 0.0
        )

    return MotionAnalysis(
        displacements=displacements,
        motion_variance=motion_variance,
        acf=acf,
        periodicity=periodicity,
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(
    rppg: Optional[RPPGAnalysis],
    motion: Optional[MotionAnalysis],
    rppg_snr_threshold: float = DEFAULT_RPPG_SNR_THRESHOLD,
    motion_var_min: float = DEFAULT_MOTION_VAR_MIN,
    periodicity_max: float = DEFAULT_PERIODICITY_MAX,
) -> Verdict:
    reasons: list[str] = []

    if rppg is None:
        flag_pulse = True
        reasons.append("rPPG: not enough usable frames to extract a pulse")
    else:
        flag_pulse = rppg.snr < rppg_snr_threshold
        if flag_pulse:
            reasons.append(
                f"rPPG SNR {rppg.snr:.2f} < threshold "
                f"{rppg_snr_threshold:.2f} (no detectable heart-rate signal — "
                f"static photo or very still subject)"
            )

    if motion is None:
        flag_motion = True
        flag_periodicity = False
        reasons.append("motion: not enough usable frames to compute displacement")
    else:
        flag_motion = motion.motion_variance < motion_var_min
        if flag_motion:
            reasons.append(
                f"motion variance {motion.motion_variance:.3f} < threshold "
                f"{motion_var_min:.2f} (face too still — likely static photo)"
            )
        flag_periodicity = motion.periodicity > periodicity_max
        if flag_periodicity:
            reasons.append(
                f"motion periodicity {motion.periodicity:.2f} > threshold "
                f"{periodicity_max:.2f} (motion is periodic — likely a "
                f"looped video replay)"
            )

    return Verdict(
        spoof=bool(reasons),
        reasons=reasons,
        flag_pulse=flag_pulse,
        flag_motion=flag_motion,
        flag_periodicity=flag_periodicity,
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render() -> None:
    st.set_page_config(
        page_title="Layer 6 — Video Liveness",
        page_icon="❤️",
        layout="wide",
    )
    st.title("Layer 6 · Video Liveness (rPPG + Micro-movement)")
    st.caption(
        "Multi-frame liveness checks: pulse extraction + motion analysis "
        f"from ~{DEFAULT_CAPTURE_SECONDS:.0f} s of video."
    )

    with st.expander("How it works", expanded=False):
        st.markdown(
            f"""
            We capture **~{DEFAULT_CAPTURE_SECONDS:.0f} seconds** of video
            at ~{DEFAULT_TARGET_FPS:.0f} fps, run per-frame face detection,
            extract a cheek-ROI mean RGB triple per frame, then compute
            three flags. ANY flag fires → SPOOF.

            1. **rPPG SNR** — CHROM-method (de Haan & Jeanne 2013) pulse
               extraction from the cheek-RGB time series, bandpassed to
               {RPPG_BAND_HZ[0]}–{RPPG_BAND_HZ[1]} Hz
               ({int(RPPG_BAND_HZ[0]*60)}–{int(RPPG_BAND_HZ[1]*60)} bpm).
               SNR = peak-band-power / median-band-power. Real cheek →
               clear peak; a photo has no pulse signal at all.

            2. **Motion variance** — frame-to-frame displacement of the
               nose+mouth midpoint. Real face: small irregular motion
               from breathing and microsaccades. Static photo: ~0.

            3. **Motion periodicity** — max absolute autocorrelation of
               the displacement series at lag > 0.5 s. Real face:
               irregular → low ACF. Looped video replay: motion repeats
               → high ACF.

            **Tips for capture:** sit ~50 cm from the camera with even
            lighting on the cheek. Don't move your head deliberately;
            just look at the screen. Avoid backlight that washes out the
            cheek region.
            """
        )

    # ----- Sidebar -----
    with st.sidebar:
        st.header("Capture")
        duration = st.slider(
            "Capture duration (s)",
            min_value=3.0,
            max_value=20.0,
            value=DEFAULT_CAPTURE_SECONDS,
            step=1.0,
        )
        target_fps = st.slider(
            "Target FPS",
            min_value=10.0,
            max_value=60.0,
            value=DEFAULT_TARGET_FPS,
            step=1.0,
            help="Higher = better pulse signal but slower per-frame "
                 "face detection. 30 fps is the sweet spot.",
        )
        min_face_conf = st.slider(
            "Face confidence",
            min_value=0.1,
            max_value=0.9,
            value=DEFAULT_MIN_FACE_CONFIDENCE,
            step=0.05,
        )

        st.divider()
        st.subheader("Thresholds")
        rppg_thresh = st.slider(
            "rPPG SNR ≥",
            min_value=0.5,
            max_value=20.0,
            value=DEFAULT_RPPG_SNR_THRESHOLD,
            step=0.5,
        )
        motion_thresh = st.slider(
            "Motion variance ≥ (px²)",
            min_value=0.0,
            max_value=10.0,
            value=DEFAULT_MOTION_VAR_MIN,
            step=0.1,
        )
        periodicity_thresh = st.slider(
            "Motion periodicity ≤",
            min_value=0.1,
            max_value=0.9,
            value=DEFAULT_PERIODICITY_MAX,
            step=0.05,
        )

    # ----- Capture controls -----
    col_btn, col_clear = st.columns([3, 1])
    do_capture = col_btn.button("🎥  Start Capture", type="primary")
    if "capture_result" in st.session_state:
        if col_clear.button("Clear capture"):
            st.session_state.pop("capture_result", None)
            st.rerun()

    preview_slot = st.empty()

    if do_capture:
        progress = st.progress(0.0, text="Opening camera…")
        try:
            result = capture_video(
                duration_sec=float(duration),
                target_fps=float(target_fps),
                min_face_conf=float(min_face_conf),
                progress_cb=lambda p, msg: progress.progress(p, text=msg),
                preview_cb=lambda f: preview_slot.image(
                    f, channels="RGB", use_container_width=True
                ),
            )
            st.session_state["capture_result"] = result
            progress.empty()
            st.success(
                f"Captured **{len(result.features)} frames** in "
                f"{duration:.0f} s (actual {result.actual_fps:.1f} fps), "
                f"**{result.n_face_frames}** with a face detected."
            )
        except RuntimeError as exc:
            progress.empty()
            st.error(str(exc))

    if "capture_result" not in st.session_state:
        st.info(
            "Click **🎥 Start Capture** to record. Sit still, ~50 cm from "
            "the camera with even lighting on your cheeks. Threshold "
            "sliders re-evaluate without re-capturing."
        )
        return

    result: CaptureResult = st.session_state["capture_result"]
    fps = result.actual_fps if result.actual_fps > 0 else 30.0

    preview_slot.image(
        result.last_frame,
        channels="RGB",
        caption="Last captured frame",
        use_container_width=True,
    )

    if result.n_face_frames < int(fps * 3):
        st.error(
            f"❌  Only **{result.n_face_frames}** frames had a detected "
            f"face (need ≥ {int(fps * 3)} for ≥ 3 s of usable signal). "
            "Re-capture with a clearer, front-on face."
        )
        return

    # ----- Analyse -----
    rppg = analyse_rppg(result.features, fps)
    motion = analyse_motion(result.features, fps)
    verdict = classify(
        rppg,
        motion,
        rppg_snr_threshold=float(rppg_thresh),
        motion_var_min=float(motion_thresh),
        periodicity_max=float(periodicity_thresh),
    )

    # ----- Verdict banner -----
    if verdict.spoof:
        st.error(
            f"**Verdict · SPOOF SUSPECTED**  "
            f"({len(verdict.reasons)} flag"
            f"{'s' if len(verdict.reasons) > 1 else ''} fired)"
        )
        for reason in verdict.reasons:
            st.write(f"  •  {reason}")
    else:
        st.success(
            "**Verdict · LIKELY LIVE FACE**  "
            "(pulse + non-periodic micro-movement detected)"
        )

    # ----- Metrics -----
    c1, c2, c3, c4 = st.columns(4)

    def metric_card(col, label, value, flagged, hint):
        col.metric(label, value, delta=hint, delta_color="off")
        col.markdown(
            ":red[🚩 flag fired]" if flagged else ":green[✓ within natural range]"
        )

    if rppg is not None:
        metric_card(
            c1, "rPPG SNR", f"{rppg.snr:.2f}",
            verdict.flag_pulse, f"thresh {rppg_thresh:.1f}",
        )
        c2.metric(
            "Heart rate (bpm)", f"{rppg.heart_rate_bpm:.0f}",
            delta="diagnostic only", delta_color="off",
        )
    else:
        c1.metric("rPPG SNR", "—", delta="no signal", delta_color="off")
        c2.metric("Heart rate (bpm)", "—", delta="—", delta_color="off")

    if motion is not None:
        metric_card(
            c3, "Motion variance (px²)", f"{motion.motion_variance:.3f}",
            verdict.flag_motion, f"min {motion_thresh:.2f}",
        )
        metric_card(
            c4, "Motion periodicity", f"{motion.periodicity:.3f}",
            verdict.flag_periodicity, f"max {periodicity_thresh:.2f}",
        )
    else:
        c3.metric("Motion variance (px²)", "—")
        c4.metric("Motion periodicity", "—")

    st.caption(
        f"Capture: {len(result.features)} frames @ {fps:.1f} fps · "
        f"{result.n_face_frames} with face detected"
    )

    # ----- Pulse charts -----
    st.divider()
    col_pulse, col_motion = st.columns(2)

    if rppg is not None:
        col_pulse.subheader("Pulse signal (CHROM, bandpassed)")
        ts = np.arange(len(rppg.pulse_signal)) / fps
        pulse_df = pd.DataFrame(
            {"pulse": rppg.pulse_signal},
            index=pd.Index(ts, name="time (s)"),
        )
        col_pulse.line_chart(pulse_df, height=240)
        col_pulse.caption(
            f"Heart rate estimate: **{rppg.heart_rate_bpm:.0f} bpm** "
            f"({rppg.heart_rate_hz:.2f} Hz) · SNR = {rppg.snr:.2f}"
        )

    if motion is not None:
        col_motion.subheader("Frame-to-frame displacement (px)")
        ts = np.arange(len(motion.displacements)) / fps
        m_df = pd.DataFrame(
            {"displacement": motion.displacements},
            index=pd.Index(ts, name="time (s)"),
        )
        col_motion.line_chart(m_df, height=240)
        col_motion.caption(
            f"Motion variance: {motion.motion_variance:.3f} px² · "
            f"max |ACF| at lag > 0.5 s: {motion.periodicity:.3f}"
        )

    # ----- FFT + ACF charts -----
    if rppg is not None:
        st.subheader("Pulse FFT")
        in_band = (rppg.fft_freqs >= 0) & (rppg.fft_freqs <= 5.0)
        fft_df = pd.DataFrame(
            {"power": rppg.fft_power[in_band]},
            index=pd.Index(rppg.fft_freqs[in_band], name="frequency (Hz)"),
        )
        st.area_chart(fft_df, height=200)
        st.caption(
            f"Heart-rate band: {RPPG_BAND_HZ[0]}–{RPPG_BAND_HZ[1]} Hz "
            f"({int(RPPG_BAND_HZ[0]*60)}–{int(RPPG_BAND_HZ[1]*60)} bpm). "
            "Strong narrow peak inside the band → live pulse."
        )

    if motion is not None:
        st.subheader("Motion autocorrelation")
        max_lag = min(len(motion.acf) - 1, int(fps * 5))
        lag_secs = np.arange(max_lag) / fps
        acf_df = pd.DataFrame(
            {"|ACF|": np.abs(motion.acf[:max_lag])},
            index=pd.Index(lag_secs, name="lag (s)"),
        )
        st.area_chart(acf_df, height=200)
        st.caption(
            "Peaks at non-zero lag indicate periodic motion → looped "
            "video replay attack."
        )


if __name__ == "__main__":
    render()
