"""
Layer 6: Video Liveness (rPPG + Micro-movement)
================================================

Concept
-------
Multi-frame PAD. Real faces have an invisible pulse and subtle micro-movements.
Spoofs (static photos, looped videos) do not.

Fix implemented: Bounded motion variance. Micro-movement must be tiny. Gross
shaking (like a vibrating hand holding a phone/photo) will now trigger a SPOOF.

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
RPPG_BAND_HZ = (0.7, 4.0)   

# ADJUSTED FOR EDGE CASES: Tighter SNR to reject hand-tremor noise.
DEFAULT_RPPG_SNR_THRESHOLD = 4.5
DEFAULT_MOTION_VAR_MIN = 0.2
# NEW: Max motion limit. Real micro-movements are tiny. Shaking paper fails this.
DEFAULT_MOTION_VAR_MAX = 3.5 
DEFAULT_PERIODICITY_MAX = 0.40
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
    flag_motion_low: bool
    flag_motion_high: bool
    flag_periodicity: bool


# ---------------------------------------------------------------------------
# Face detection helpers 
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
    eye = kps["right_eye"]
    mouth = kps["mouth_center"]
    cx, cy = (eye[0] + mouth[0]) / 2, (eye[1] + mouth[1]) / 2
    bx, by, bw, bh = bbox
    
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
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

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
                progress_cb(min(now / duration_sec, 1.0), f"{len(features)} frames")
            if preview_cb is not None and len(features) % 4 == 0:
                preview_cb(frame_rgb)
    finally:
        cap.release()

    total = time.perf_counter() - start_t
    actual_fps = len(features) / total if total > 0 else 0.0

    return CaptureResult(
        features=features,
        actual_fps=actual_fps,
        last_frame=last_frame_rgb if last_frame_rgb is not None else np.zeros((100, 100, 3), dtype=np.uint8),
        n_face_frames=n_with_face,
    )


# ---------------------------------------------------------------------------
# rPPG via the CHROM algorithm
# ---------------------------------------------------------------------------

def chrom_pulse(rgb_signal: np.ndarray, fps: float):
    N = rgb_signal.shape[0]
    if N < int(fps * 3):
        return np.zeros(0), np.zeros(0), np.zeros(0), 0.0, 0.0

    mean = rgb_signal.mean(axis=0)
    if (mean <= 0).any():
        return np.zeros(N), np.zeros(0), np.zeros(0), 0.0, 0.0
    norm = rgb_signal / mean - 1.0

    R, G, B = norm[:, 0], norm[:, 1], norm[:, 2]
    X = 3.0 * R - 2.0 * G
    Y = 1.5 * R + G - 1.5 * B

    nyq = fps / 2.0
    low = RPPG_BAND_HZ[0] / nyq
    high = min(RPPG_BAND_HZ[1] / nyq, 0.99)
    if low <= 0 or high <= low or high >= 1.0:
        return np.zeros(N), np.zeros(0), np.zeros(0), 0.0, 0.0
    try:
        b, a = butter(3, [low, high], btype="band")
        padlen = 3 * max(len(a), len(b))
        if N <= padlen + 1:
            return np.zeros(N), np.zeros(0), np.zeros(0), 0.0, 0.0
        X_bp = filtfilt(b, a, X)
        Y_bp = filtfilt(b, a, Y)
    except Exception:
        return np.zeros(N), np.zeros(0), np.zeros(0), 0.0, 0.0

    sy = float(np.std(Y_bp))
    alpha = float(np.std(X_bp)) / sy if sy > 1e-9 else 1.0
    pulse = X_bp - alpha * Y_bp

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
# Micro-movement
# ---------------------------------------------------------------------------

def analyse_motion(features: list, fps: float) -> Optional[MotionAnalysis]:
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
        periodicity = float(np.max(np.abs(acf[min_lag:max_lag]))) if max_lag > min_lag else 0.0

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
    rppg_snr_threshold: float,
    motion_var_min: float,
    motion_var_max: float,
    periodicity_max: float,
) -> Verdict:
    reasons: list[str] = []
    
    flag_pulse = False
    if rppg is None:
        flag_pulse = True
        reasons.append("rPPG: not enough usable frames to extract a pulse")
    else:
        flag_pulse = rppg.snr < rppg_snr_threshold
        if flag_pulse:
            reasons.append(f"rPPG SNR {rppg.snr:.2f} < threshold {rppg_snr_threshold:.2f} (no detectable heart rate)")

    flag_motion_low = False
    flag_motion_high = False
    flag_periodicity = False

    if motion is None:
        flag_motion_low = True
        reasons.append("motion: not enough usable frames")
    else:
        flag_motion_low = motion.motion_variance < motion_var_min
        flag_motion_high = motion.motion_variance > motion_var_max
        
        if flag_motion_low:
            reasons.append(f"Motion var {motion.motion_variance:.3f} < {motion_var_min:.2f} (too still)")
        if flag_motion_high:
            reasons.append(f"Motion var {motion.motion_variance:.3f} > {motion_var_max:.2f} (gross movement/shaking detected)")

        flag_periodicity = motion.periodicity > periodicity_max
        if flag_periodicity:
            reasons.append(f"Periodicity {motion.periodicity:.2f} > {periodicity_max:.2f} (looped motion detected)")

    return Verdict(
        spoof=bool(reasons),
        reasons=reasons,
        flag_pulse=flag_pulse,
        flag_motion_low=flag_motion_low,
        flag_motion_high=flag_motion_high,
        flag_periodicity=flag_periodicity,
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render() -> None:
    st.set_page_config(page_title="Layer 6 — Video Liveness", page_icon="❤️", layout="wide")
    st.title("Layer 6 · Video Liveness (rPPG + Micro-movement)")

    with st.sidebar:
        st.header("Capture")
        duration = st.slider("Capture duration (s)", 3.0, 20.0, DEFAULT_CAPTURE_SECONDS, 1.0)
        target_fps = st.slider("Target FPS", 10.0, 60.0, DEFAULT_TARGET_FPS, 1.0)
        min_face_conf = st.slider("Face conf", 0.1, 0.9, DEFAULT_MIN_FACE_CONFIDENCE, 0.05)

        st.divider()
        st.subheader("Thresholds")
        rppg_thresh = st.slider("rPPG SNR ≥", 0.5, 20.0, DEFAULT_RPPG_SNR_THRESHOLD, 0.5)
        motion_thresh_min = st.slider("Min Motion Var (px²)", 0.0, 2.0, DEFAULT_MOTION_VAR_MIN, 0.1)
        motion_thresh_max = st.slider("Max Motion Var (px²)", 1.0, 15.0, DEFAULT_MOTION_VAR_MAX, 0.5)
        periodicity_thresh = st.slider("Periodicity ≤", 0.1, 0.9, DEFAULT_PERIODICITY_MAX, 0.05)

    col_btn, col_clear = st.columns([3, 1])
    do_capture = col_btn.button("🎥  Start Capture", type="primary")
    if "capture_result" in st.session_state and col_clear.button("Clear capture"):
        st.session_state.pop("capture_result", None)
        st.rerun()

    preview_slot = st.empty()

    if do_capture:
        progress = st.progress(0.0, text="Capturing...")
        try:
            result = capture_video(
                float(duration), float(target_fps), float(min_face_conf),
                lambda p, msg: progress.progress(p, text=msg),
                lambda f: preview_slot.image(f, channels="RGB", use_container_width=True),
            )
            st.session_state["capture_result"] = result
            progress.empty()
        except RuntimeError as exc:
            progress.empty()
            st.error(str(exc))

    if "capture_result" not in st.session_state:
        return

    result: CaptureResult = st.session_state["capture_result"]
    fps = result.actual_fps if result.actual_fps > 0 else 30.0

    preview_slot.image(result.last_frame, channels="RGB", use_container_width=True)

    if result.n_face_frames < int(fps * 3):
        st.error("❌ Not enough face frames. Re-capture.")
        return

    rppg = analyse_rppg(result.features, fps)
    motion = analyse_motion(result.features, fps)
    verdict = classify(
        rppg, motion, float(rppg_thresh), float(motion_thresh_min), float(motion_thresh_max), float(periodicity_thresh)
    )

    if verdict.spoof:
        st.error(f"**Verdict · SPOOF SUSPECTED** ({len(verdict.reasons)} flags fired)")
        for r in verdict.reasons:
            st.write(f"  •  {r}")
    else:
        st.success("**Verdict · LIKELY LIVE FACE** (pulse + natural micro-movement detected)")

    c1, c2, c3, c4 = st.columns(4)
    def metric_card(col, label, value, flagged, hint):
        col.metric(label, value, delta=hint, delta_color="off")
        col.markdown(":red[🚩 flag fired]" if flagged else ":green[✓ valid]")

    if rppg:
        metric_card(c1, "rPPG SNR", f"{rppg.snr:.2f}", verdict.flag_pulse, f"thresh {rppg_thresh:.1f}")
        c2.metric("Heart rate", f"{rppg.heart_rate_bpm:.0f} bpm", delta="diagnostic", delta_color="off")
    
    if motion:
        flagged_motion = verdict.flag_motion_low or verdict.flag_motion_high
        metric_card(c3, "Motion Variance", f"{motion.motion_variance:.3f}", flagged_motion, f"bounds {motion_thresh_min:.1f}-{motion_thresh_max:.1f}")
        metric_card(c4, "Periodicity", f"{motion.periodicity:.3f}", verdict.flag_periodicity, f"max {periodicity_thresh:.2f}")

    st.divider()
    col_pulse, col_motion = st.columns(2)
    if rppg:
        col_pulse.subheader("Pulse signal")
        col_pulse.line_chart(pd.DataFrame({"pulse": rppg.pulse_signal}, index=np.arange(len(rppg.pulse_signal))/fps), height=200)
    if motion:
        col_motion.subheader("Frame-to-frame displacement")
        col_motion.line_chart(pd.DataFrame({"displacement": motion.displacements}, index=np.arange(len(motion.displacements))/fps), height=200)

if __name__ == "__main__":
    render()