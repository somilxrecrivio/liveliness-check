"""
Layer 3: Spatial Passive PAD
=============================

Concept
-------
Differentiate a *live 3-D face* from a *2-D presentation attack*
(printed photo, displayed screen) using two cheap spatial-domain
heuristics computed on the detected face ROI:

1. **LBP (Local Binary Pattern) code variance**
       Real skin has rich micro-texture — pores, hair follicles, fine
       wrinkles — that produces many distinct LBP codes across the
       ROI. 

2. **Laplacian variance** of the grayscale ROI
       The classical sharpness/blur metric (Pech-Pacheco 2000). Real
       in-focus skin has many high-amplitude Laplacian responses
       around pores and edges.

Both metrics are **"higher = more real"**. The verdict fires when
*either* metric falls below its threshold, biasing toward false
positives (better to make a real user re-shoot than to admit a
spoofed identity).

Run with:
    streamlit run layer3_passive_pad.py
"""

import os
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from skimage import feature

# MediaPipe 0.10.x exposes BaseOptions / FaceDetector as attributes on
# mp.tasks but not as importable names — alias them locally.
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_IMAGE_SIDE = 1024
FACE_CROP_SIZE = 256
CHEEK_CROP_SIZE = 96

LBP_N_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"
LBP_N_BINS = LBP_N_POINTS + 2

# ADJUSTED FOR EDGE CASES: Lowered LBP threshold to 1.5 to accommodate 
# aggressive ISP denoising on modern smartphones.
DEFAULT_LBP_VAR_THRESHOLD = 1.5          
# ADJUSTED FOR EDGE CASES: Lowered to 40.0 to handle poor network compression
DEFAULT_LAPLACIAN_THRESHOLD = 40.0       
DEFAULT_MOIRE_PEAK_THRESHOLD = 12        
DEFAULT_MIN_FACE_CONFIDENCE = 0.5

# Moiré FFT-annulus parameters
MOIRE_INNER_FRAC = 0.25
MOIRE_OUTER_FRAC = 0.85
MOIRE_PEAK_SIGMA = 3.5

# ADJUSTED FOR EDGE CASES: Widened IQA gates to prevent demographic bias 
# (darker skin tones) and variable lighting rejections.
DEFAULT_IQA_BRIGHTNESS_MIN = 30.0    # lowered from 40
DEFAULT_IQA_BRIGHTNESS_MAX = 240.0   # raised from 235
DEFAULT_IQA_CONTRAST_MIN = 12.0      # lowered from 18
DEFAULT_IQA_DYNAMIC_RANGE_MIN = 40.0 # lowered from 50
DEFAULT_IQA_NOISE_MAX = 25.0         # raised from 22


@dataclass
class IQAResult:
    brightness: float
    contrast: float
    dynamic_range: float
    noise_estimate: float
    quality_ok: bool
    quality_issues: list


@dataclass
class MoireAnalysis:
    log_spectrum: np.ndarray
    peak_mask: np.ndarray
    peak_count: int


@dataclass
class TextureAnalysis:
    lbp_image: np.ndarray
    lbp_histogram: np.ndarray
    lbp_code_variance: float
    laplacian_image: np.ndarray
    laplacian_variance: float
    moire: MoireAnalysis
    iqa: IQAResult


@dataclass
class Verdict:
    spoof: bool
    quality_ok: bool
    quality_issues: list
    reasons: list
    flag_lbp: bool
    flag_lap: bool
    flag_moire: bool


# ---------------------------------------------------------------------------
# Face detection (MediaPipe Tasks API — BlazeFace short-range)
# ---------------------------------------------------------------------------

BLAZEFACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BLAZEFACE_MODEL_PATH = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")

KEYPOINT_NAMES = [
    "right_eye",
    "left_eye",
    "nose_tip",
    "mouth_center",
    "right_ear_tragion",
    "left_ear_tragion",
]

_FACE_DETECTOR = None
_FACE_DETECTOR_CONF: float | None = None


def _ensure_model_downloaded(path: str = BLAZEFACE_MODEL_PATH) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(BLAZEFACE_MODEL_URL, path)
    return path


def _get_face_detector(min_confidence: float):
    global _FACE_DETECTOR, _FACE_DETECTOR_CONF
    if _FACE_DETECTOR is None or _FACE_DETECTOR_CONF != min_confidence:
        model_path = _ensure_model_downloaded()
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            min_detection_confidence=min_confidence,
        )
        _FACE_DETECTOR = FaceDetector.create_from_options(options)
        _FACE_DETECTOR_CONF = min_confidence
    return _FACE_DETECTOR


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]
    keypoints: dict[str, tuple[int, int]]
    confidence: float


def detect_face(
    image_rgb: np.ndarray, min_confidence: float = DEFAULT_MIN_FACE_CONFIDENCE
) -> FaceDetection | None:
    detector = _get_face_detector(min_confidence)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)
    if not result.detections:
        return None

    detections = sorted(
        result.detections,
        key=lambda d: d.bounding_box.width * d.bounding_box.height,
        reverse=True,
    )
    det = detections[0]
    h_img, w_img = image_rgb.shape[:2]
    bb = det.bounding_box

    x = max(0, int(bb.origin_x))
    y = max(0, int(bb.origin_y))
    w = max(1, min(int(bb.width), w_img - x))
    h = max(1, min(int(bb.height), h_img - y))

    keypoints = {}
    for name, kp in zip(KEYPOINT_NAMES, det.keypoints):
        kx = int(kp.x * w_img) if 0.0 <= kp.x <= 1.0 else int(kp.x)
        ky = int(kp.y * h_img) if 0.0 <= kp.y <= 1.0 else int(kp.y)
        keypoints[name] = (kx, ky)

    confidence = float(det.categories[0].score) if det.categories else 0.0

    return FaceDetection(
        bbox=(x, y, w, h),
        keypoints=keypoints,
        confidence=confidence,
    )


def crop_face(
    image_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_size: int = FACE_CROP_SIZE,
    margin_frac: float = 0.10,
) -> np.ndarray:
    x, y, w, h = bbox
    h_img, w_img = image_rgb.shape[:2]
    margin = int(margin_frac * min(w, h))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w_img, x + w + margin)
    y1 = min(h_img, y + h + margin)
    crop = image_rgb[y0:y1, x0:x1]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)


def crop_cheek(
    image_rgb: np.ndarray,
    detection: FaceDetection,
    target_size: int = CHEEK_CROP_SIZE,
) -> np.ndarray | None:
    """
    Dynamically extract the cleanest cheek patch.
    Calculates distance from nose to both eyes to find the side of the 
    face most exposed to the camera, handling severe side profiles.
    """
    h_img, w_img = image_rgb.shape[:2]
    r_eye = detection.keypoints["right_eye"]
    l_eye = detection.keypoints["left_eye"]
    nose = detection.keypoints["nose_tip"]
    mouth = detection.keypoints["mouth_center"]

    # Determine most frontal cheek based on horizontal distance from nose to eye
    dist_r = abs(nose[0] - r_eye[0])
    dist_l = abs(nose[0] - l_eye[0])
    best_eye = r_eye if dist_r > dist_l else l_eye

    cx = (best_eye[0] + mouth[0]) // 2
    cy = (best_eye[1] + mouth[1]) // 2
    
    bx, by, bw, bh = detection.bbox
    if cx < bx + bw // 2:
        cx = (cx + bx) // 2
    else:
        cx = (cx + bx + bw) // 2

    half = max(8, min(bw, bh) // 8)
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w_img, cx + half)
    y1 = min(h_img, cy + half)

    if x1 - x0 < 12 or y1 - y0 < 12:
        return None

    patch = image_rgb[y0:y1, x0:x1]
    # return cv2.resize(patch, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return patch


# ---------------------------------------------------------------------------
# Texture features
# ---------------------------------------------------------------------------

def compute_lbp(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    lbp = feature.local_binary_pattern(
        gray, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD
    )
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=LBP_N_BINS,
        range=(0, LBP_N_BINS),
        density=False,
    )
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    code_variance = float(np.var(lbp.ravel()))
    return lbp, hist, code_variance


def compute_laplacian(gray: np.ndarray) -> tuple[np.ndarray, float]:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap, float(lap.var())


def to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim == 2:
        return image_rgb.astype(np.uint8)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def compute_iqa(
    gray: np.ndarray,
    brightness_min: float = DEFAULT_IQA_BRIGHTNESS_MIN,
    brightness_max: float = DEFAULT_IQA_BRIGHTNESS_MAX,
    contrast_min: float = DEFAULT_IQA_CONTRAST_MIN,
    dynamic_range_min: float = DEFAULT_IQA_DYNAMIC_RANGE_MIN,
    noise_max: float = DEFAULT_IQA_NOISE_MAX,
) -> IQAResult:
    gray_f = gray.astype(np.float32)
    brightness = float(np.mean(gray_f))
    contrast = float(np.std(gray_f))
    p5, p95 = np.percentile(gray_f, [5, 95])
    dynamic_range = float(p95 - p5)

    median = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    noise_estimate = float(np.std(gray_f - median))

    issues: list[str] = []
    if brightness < brightness_min:
        issues.append(f"Image too dark (brightness {brightness:.0f} < {brightness_min:.0f})")
    if brightness > brightness_max:
        issues.append(f"Image blown out (brightness {brightness:.0f} > {brightness_max:.0f})")
    if contrast < contrast_min:
        issues.append(f"Flat lighting (contrast {contrast:.1f} < {contrast_min:.0f})")
    if dynamic_range < dynamic_range_min:
        issues.append(f"Narrow dynamic range ({dynamic_range:.0f} < {dynamic_range_min:.0f})")
    if noise_estimate > noise_max:
        issues.append(f"Motion blur or severe noise detected (noise {noise_estimate:.1f} > {noise_max:.0f})")

    return IQAResult(
        brightness=brightness,
        contrast=contrast,
        dynamic_range=dynamic_range,
        noise_estimate=noise_estimate,
        quality_ok=not issues,
        quality_issues=issues,
    )


def compute_moire(
    gray: np.ndarray,
    inner_frac: float = MOIRE_INNER_FRAC,
    outer_frac: float = MOIRE_OUTER_FRAC,
    peak_sigma: float = MOIRE_PEAK_SIGMA,
) -> MoireAnalysis:
    h, w = gray.shape
    window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    f = np.fft.fft2(gray.astype(np.float32) * window)
    log_mag = np.log1p(np.abs(np.fft.fftshift(f)))

    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    r_norm = r / r.max()
    band = (r_norm > inner_frac) & (r_norm < outer_frac)

    in_band = log_mag[band]
    thresh = float(in_band.mean() + peak_sigma * in_band.std())
    peak_mask = (log_mag > thresh) & band
    return MoireAnalysis(
        log_spectrum=log_mag,
        peak_mask=peak_mask,
        peak_count=int(peak_mask.sum()),
    )


def analyse(face_rgb: np.ndarray) -> TextureAnalysis:
    gray = to_grayscale(face_rgb)
    lbp_img, lbp_hist, lbp_code_var = compute_lbp(gray)
    lap_img, lap_var = compute_laplacian(gray)
    moire = compute_moire(gray)
    iqa = compute_iqa(gray)
    return TextureAnalysis(
        lbp_image=lbp_img.astype(np.uint8),
        lbp_histogram=lbp_hist,
        lbp_code_variance=lbp_code_var,
        laplacian_image=lap_img,
        laplacian_variance=lap_var,
        moire=moire,
        iqa=iqa,
    )


def classify(
    analysis: TextureAnalysis,
    lbp_threshold: float,
    laplacian_threshold: float,
    moire_threshold: int = DEFAULT_MOIRE_PEAK_THRESHOLD,
) -> Verdict:
    if not analysis.iqa.quality_ok:
        return Verdict(
            spoof=False,
            quality_ok=False,
            quality_issues=list(analysis.iqa.quality_issues),
            reasons=[],
            flag_lbp=False,
            flag_lap=False,
            flag_moire=False,
        )

    flag_lbp = analysis.lbp_code_variance < lbp_threshold
    flag_lap = analysis.laplacian_variance < laplacian_threshold
    flag_moire = analysis.moire.peak_count > moire_threshold

    reasons: list[str] = []
    if flag_lbp:
        reasons.append(
            f"LBP code variance {analysis.lbp_code_variance:.3f} "
            f"< threshold {lbp_threshold:.3f} (texture too uniform)"
        )
    if flag_lap:
        reasons.append(
            f"Laplacian variance {analysis.laplacian_variance:.1f} "
            f"< threshold {laplacian_threshold:.1f} (image too blurry)"
        )
    if flag_moire:
        reasons.append(
            f"Moiré peak count {analysis.moire.peak_count} "
            f"> threshold {moire_threshold} (halftone/screen pattern detected)"
        )

    return Verdict(
        spoof=bool(reasons),
        quality_ok=True,
        quality_issues=[],
        reasons=reasons,
        flag_lbp=flag_lbp,
        flag_lap=flag_lap,
        flag_moire=flag_moire,
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def lbp_to_color_image(lbp: np.ndarray) -> np.ndarray:
    lo, hi = float(lbp.min()), float(lbp.max())
    if hi - lo < 1e-12:
        normalised = np.zeros_like(lbp, dtype=np.uint8)
    else:
        normalised = ((lbp - lo) / (hi - lo) * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(normalised, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


def laplacian_to_image(lap: np.ndarray) -> np.ndarray:
    a = np.abs(lap)
    if a.max() > 0:
        a = (a / a.max() * 255).astype(np.uint8)
    else:
        a = a.astype(np.uint8)
    return a


def moire_spectrum_image(moire: MoireAnalysis) -> np.ndarray:
    log_mag = moire.log_spectrum
    lo, hi = float(log_mag.min()), float(log_mag.max())
    if hi - lo < 1e-12:
        norm = np.zeros_like(log_mag, dtype=np.uint8)
    else:
        norm = ((log_mag - lo) / (hi - lo) * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    rgb = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(moire.peak_mask.astype(np.uint8), kernel, iterations=1)
    rgb[dilated > 0] = [0, 255, 0]
    return rgb


def draw_face_overlay(
    image_rgb: np.ndarray,
    detection: FaceDetection,
    spoof: bool,
) -> np.ndarray:
    out = image_rgb.copy()
    x, y, w, h = detection.bbox
    color = (220, 60, 60) if spoof else (60, 200, 60)
    thickness = max(2, int(min(w, h) * 0.012))
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)

    for name, (kx, ky) in detection.keypoints.items():
        cv2.circle(out, (kx, ky), max(2, thickness), color, -1)

    label = "SPOOF" if spoof else "REAL"
    font_scale = max(0.5, min(w, h) * 0.004)
    cv2.putText(
        out,
        f"{label}  conf={detection.confidence:.2f}",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        max(1, int(thickness * 0.6)),
        cv2.LINE_AA,
    )
    return out


# ---------------------------------------------------------------------------
# Image prep
# ---------------------------------------------------------------------------

def prepare_image(image_rgb: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if max(h, w) <= max_side:
        return image_rgb
    scale = max_side / max(h, w)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render() -> None:
    st.set_page_config(
        page_title="Layer 3 — Spatial Passive PAD",
        page_icon="🧪",
        layout="wide",
    )
    st.title("Layer 3 · Spatial Passive PAD")
    st.caption(
        "Differentiate live skin from printed photos and screen replay "
        "using LBP texture and Laplacian sharpness on the face ROI."
    )

    with st.expander("How the detection works", expanded=False):
        st.markdown(
            f"""
            **Quality gate first**, then **three flags vote — any flag → SPOOF.**

            **IQA quality gate**: Out-of-range frames ask for a retake instead 
            of producing a bogus verdict.

            1. **LBP code variance** — variance of the per-pixel uniform LBP codes.
            2. **Laplacian variance** — Pech-Pacheco focus measure.
            3. **Moiré peak count** — FFT-based.

            ROI source: **MediaPipe** BlazeFace. The **cheek patch** dynamically 
            selects the cheek most exposed to the camera to avoid glasses, 
            hair, and extreme side angles.
            """
        )

    with st.sidebar:
        st.header("Image source")
        source = st.radio(
            "Source",
            ("Live camera (st.camera_input)", "Upload file"),
            label_visibility="collapsed",
        )

        st.divider()
        st.subheader("Detection thresholds")
        st.caption("ANY flag fires → SPOOF verdict")
        lbp_thresh = st.number_input(
            "LBP code variance threshold",
            min_value=0.0,
            value=DEFAULT_LBP_VAR_THRESHOLD,
            step=0.1,
            format="%.2f",
        )
        lap_thresh = st.number_input(
            "Laplacian variance threshold",
            min_value=0.0,
            value=DEFAULT_LAPLACIAN_THRESHOLD,
            step=5.0,
            format="%.1f",
        )
        moire_thresh = st.number_input(
            "Moiré peak count threshold",
            min_value=0,
            value=DEFAULT_MOIRE_PEAK_THRESHOLD,
            step=1,
        )
        min_face_conf = st.slider(
            "MediaPipe face confidence",
            min_value=0.1,
            max_value=0.9,
            value=DEFAULT_MIN_FACE_CONFIDENCE,
            step=0.05,
        )

        st.divider()
        st.subheader("Analysis target")
        use_cheek = st.checkbox(
            "Use cheek patch (recommended)",
            value=True,
        )

    pil_image: Image.Image | None = None
    if source.startswith("Live"):
        snap = st.camera_input("Take a snapshot of your face")
        if snap is not None:
            pil_image = Image.open(snap)
    else:
        upload = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        if upload is not None:
            pil_image = Image.open(upload)

    if pil_image is None:
        st.info("Awaiting input.")
        return

    pil_image = ImageOps.exif_transpose(pil_image)
    rgb_full = np.asarray(pil_image.convert("RGB"))
    rgb_full = prepare_image(rgb_full, MAX_IMAGE_SIDE)

    detection = detect_face(rgb_full, min_confidence=float(min_face_conf))

    if detection is None:
        st.error(
            "We couldn't clearly see your face. Please ensure you are looking at the camera."
        )
        st.image(rgb_full, use_container_width=True)
        return

    face_rgb = crop_face(rgb_full, detection.bbox, FACE_CROP_SIZE)
    cheek_rgb = crop_cheek(rgb_full, detection, CHEEK_CROP_SIZE) if use_cheek else None

    analysis_roi = cheek_rgb if (cheek_rgb is not None and use_cheek) else face_rgb
    analysis = analyse(analysis_roi)
    verdict = classify(
        analysis,
        lbp_threshold=lbp_thresh,
        laplacian_threshold=lap_thresh,
        moire_threshold=int(moire_thresh),
    )

    if cheek_rgb is None and use_cheek:
        st.warning("Could not extract a clean cheek patch. Falling back to full face.")

    if not verdict.quality_ok:
        st.warning(
            f"⚠️  **We couldn't verify this image. Please adjust and try again.**"
        )
        for issue in verdict.quality_issues:
            st.write(f"  •  {issue}")
    elif verdict.spoof:
        st.error(f"**Verdict · SPOOF SUSPECTED** ({len(verdict.reasons)} flags fired)")
        for reason in verdict.reasons:
            st.write(f"  •  {reason}")
    else:
        st.success("**Verdict · LIKELY REAL FACE** (all flags passed)")

    c1, c2, c3, c4 = st.columns(4)

    def metric_card(col, label, value, flagged, hint):
        col.metric(label, value, delta=hint, delta_color="off")
        col.markdown(":red[🚩 flag fired]" if flagged else ":green[✓ valid]")

    metric_card(c1, "LBP variance", f"{analysis.lbp_code_variance:.3f}", verdict.flag_lbp, f"thresh {lbp_thresh:.2f}")
    metric_card(c2, "Laplacian", f"{analysis.laplacian_variance:.1f}", verdict.flag_lap, f"thresh {lap_thresh:.1f}")
    metric_card(c3, "Moiré peaks", f"{analysis.moire.peak_count}", verdict.flag_moire, f"thresh {int(moire_thresh)}")
    c4.metric("Face conf", f"{detection.confidence:.2f}", delta="diagnostic", delta_color="off")

    st.caption("Image Quality (gate)")
    iq1, iq2, iq3, iq4 = st.columns(4)
    iqa = analysis.iqa

    def iqa_card(col, label, value, ok, hint):
        col.metric(label, value, delta=hint, delta_color="off")
        col.markdown(":green[✓ OK]" if ok else ":red[🚩 out of range]")

    iqa_card(iq1, "Brightness", f"{iqa.brightness:.0f}", DEFAULT_IQA_BRIGHTNESS_MIN <= iqa.brightness <= DEFAULT_IQA_BRIGHTNESS_MAX, f"{DEFAULT_IQA_BRIGHTNESS_MIN:.0f}-{DEFAULT_IQA_BRIGHTNESS_MAX:.0f}")
    iqa_card(iq2, "Contrast", f"{iqa.contrast:.1f}", iqa.contrast >= DEFAULT_IQA_CONTRAST_MIN, f"≥ {DEFAULT_IQA_CONTRAST_MIN:.0f}")
    iqa_card(iq3, "Dynamic range", f"{iqa.dynamic_range:.0f}", iqa.dynamic_range >= DEFAULT_IQA_DYNAMIC_RANGE_MIN, f"≥ {DEFAULT_IQA_DYNAMIC_RANGE_MIN:.0f}")
    iqa_card(iq4, "Noise estimate", f"{iqa.noise_estimate:.1f}", iqa.noise_estimate <= DEFAULT_IQA_NOISE_MAX, f"≤ {DEFAULT_IQA_NOISE_MAX:.0f}")

    st.divider()
    col_in, col_face, col_cheek = st.columns(3)
    annotated = draw_face_overlay(rgb_full, detection, verdict.spoof)
    col_in.subheader("Input + keypoints")
    col_in.image(annotated, use_container_width=True)
    col_face.subheader(f"Face ROI")
    col_face.image(face_rgb, use_container_width=True)
    col_cheek.subheader("Cheek patch" if use_cheek and cheek_rgb is not None else "Cheek skipped")
    if cheek_rgb is not None:
        col_cheek.image(cheek_rgb, use_container_width=True)

    st.divider()
    col_lbp_img, col_lap_img, col_moire_img = st.columns(3)
    col_lbp_img.subheader("LBP code map")
    col_lbp_img.image(lbp_to_color_image(analysis.lbp_image), use_container_width=True)
    col_lap_img.subheader("|Laplacian|")
    col_lap_img.image(laplacian_to_image(analysis.laplacian_image), use_container_width=True)
    col_moire_img.subheader("Moiré peaks")
    col_moire_img.image(moire_spectrum_image(analysis.moire), use_container_width=True)

    st.subheader("LBP histogram")
    hist_df = pd.DataFrame(
        {"probability": analysis.lbp_histogram},
        index=pd.Index(np.arange(LBP_N_BINS), name="LBP uniform code"),
    )
    st.bar_chart(hist_df, height=240)


if __name__ == "__main__":
    render()