"""
Layer 5: Monocular Depth PAD (Perimeter Variance Update)
=============================

Concept
-------
A real face has visible 3-D structure. A 2-D presentation attack doesn't.
To defeat high-resolution photos where the model "hallucinates" depth,
we check the variance of the perimeter of the face box. Real faces have
background, shoulders, and chin on the perimeter (high variance). Photos
and screens have uniform flat surfaces on the perimeter (low variance).

Run:
    streamlit run layer5_depth_pad.py
"""

import os
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

DEPTH_MODEL_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small/"
    "resolve/main/onnx/model.onnx"
)
DEPTH_MODEL_PATH = os.path.join(MODELS_DIR, "depth_anything_v2_small.onnx")

BLAZEFACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
BLAZEFACE_MODEL_PATH = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")

DEPTH_INPUT_SIZE = 518
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Thresholds
DEFAULT_DEPTH_VAR_THRESHOLD = 0.010
DEFAULT_CENTRE_EDGE_THRESHOLD = 0.03
DEFAULT_PERIMETER_VAR_THRESHOLD = 0.005 # NEW: Fails flat surfaces
DEFAULT_MIN_FACE_CONFIDENCE = 0.5
BASE_FACE_CROP_MARGIN = 0.6  # Wide context

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions


@dataclass
class FaceBox:
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class DepthAnalysis:
    depth_raw: np.ndarray
    depth_normalised_full: np.ndarray
    tight_norm: np.ndarray
    tight_box: tuple[int, int, int, int]
    depth_variance: float
    centre_depth: float
    edge_depth: float
    centre_minus_edge: float
    perimeter_variance: float # NEW


@dataclass
class Verdict:
    spoof: bool
    reasons: list
    flag_var: bool
    flag_centre: bool
    flag_perimeter: bool # NEW


# ---------------------------------------------------------------------------
# Model loading 
# ---------------------------------------------------------------------------

_FACE_DETECTOR = None
_FACE_DETECTOR_CONF: float | None = None

def download_with_progress(url: str, path: str, desc: str):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    placeholder = st.empty()
    progress_bar = placeholder.progress(0, text=f"Downloading {desc}...")
    def hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(1.0, (block_num * block_size) / total_size)
            progress_bar.progress(percent, text=f"Downloading {desc}... {int(percent*100)}%")
    urllib.request.urlretrieve(url, path, reporthook=hook)
    placeholder.empty()
    return path


def _get_face_detector(min_confidence: float):
    global _FACE_DETECTOR, _FACE_DETECTOR_CONF
    if _FACE_DETECTOR is None or _FACE_DETECTOR_CONF != min_confidence:
        path = download_with_progress(BLAZEFACE_MODEL_URL, BLAZEFACE_MODEL_PATH, "Face Detector (230 KB)")
        opts = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=path),
            min_detection_confidence=min_confidence,
        )
        _FACE_DETECTOR = FaceDetector.create_from_options(opts)
        _FACE_DETECTOR_CONF = min_confidence
    return _FACE_DETECTOR


@st.cache_resource(show_spinner=False)
def get_depth_session():
    path = DEPTH_MODEL_PATH
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    return session, inp.name, DEPTH_INPUT_SIZE


# ---------------------------------------------------------------------------
# Face detection + Wide Crop
# ---------------------------------------------------------------------------

def detect_face(image_rgb: np.ndarray, min_confidence: float) -> FaceBox | None:
    detector = _get_face_detector(min_confidence)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)
    if not result.detections:
        return None
    detections = sorted(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height, reverse=True)
    det = detections[0]
    h_img, w_img = image_rgb.shape[:2]
    bb = det.bounding_box
    x = max(0, int(bb.origin_x))
    y = max(0, int(bb.origin_y))
    w = max(1, min(int(bb.width), w_img - x))
    h = max(1, min(int(bb.height), h_img - y))
    conf = float(det.categories[0].score) if det.categories else 0.0
    return FaceBox(bbox=(x, y, w, h), confidence=conf)


def crop_wide_context(image_rgb: np.ndarray, face_box: FaceBox) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = face_box.bbox
    h_img, w_img = image_rgb.shape[:2]
    m = int(BASE_FACE_CROP_MARGIN * max(w, h))
    x0 = max(0, x - m)
    y0 = max(0, y - m)
    x1 = min(w_img, x + w + m)
    y1 = min(h_img, y + h + m)
    return image_rgb[y0:y1, x0:x1], (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Depth inference + analysis
# ---------------------------------------------------------------------------

def preprocess_for_depth(crop_rgb: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(crop_rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[None]
    return arr.astype(np.float32)


def infer_depth(crop_rgb: np.ndarray, session: ort.InferenceSession, input_name: str, size: int) -> np.ndarray:
    tensor = preprocess_for_depth(crop_rgb, size)
    outputs = session.run(None, {input_name: tensor})
    depth = outputs[0]
    while depth.ndim > 2:
        depth = depth[0]
    return depth.astype(np.float32)


def _centre_edge_stats(tight_norm: np.ndarray, centre_frac: float = 0.30, edge_frac: float = 0.15) -> tuple[float, float, float]:
    h, w = tight_norm.shape
    cy, cx = h // 2, w // 2
    half_h = int(centre_frac * h / 2)
    half_w = int(centre_frac * w / 2)
    es_h = max(1, int(edge_frac * h))
    es_w = max(1, int(edge_frac * w))

    centre_patch = tight_norm[max(0, cy - half_h) : cy + half_h, max(0, cx - half_w) : cx + half_w]
    centre = float(centre_patch.mean()) if centre_patch.size else 0.0

    corners = [
        tight_norm[:es_h, :es_w].mean(),
        tight_norm[:es_h, w - es_w :].mean(),
        tight_norm[h - es_h :, :es_w].mean(),
        tight_norm[h - es_h :, w - es_w :].mean(),
    ]
    edge = float(np.mean(corners))
    
    # NEW: Calculate variance of the perimeter border
    top = tight_norm[0, :]
    bottom = tight_norm[h-1, :]
    left = tight_norm[1:h-1, 0]
    right = tight_norm[1:h-1, w-1]
    perimeter = np.concatenate([top, bottom, left, right])
    perimeter_variance = float(np.var(perimeter)) if perimeter.size else 0.0

    return centre, edge, perimeter_variance


def analyse(
    face_rgb: np.ndarray,
    crop_coords: tuple[int, int, int, int],
    face_box: FaceBox,
    session: ort.InferenceSession,
    input_name: str,
    size: int,
) -> DepthAnalysis:
    
    depth_raw = infer_depth(face_rgb, session, input_name, size)
    
    d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
    depth_range = d_max - d_min
    if depth_range < 1e-12:
        depth_norm_full = np.zeros_like(depth_raw)
    else:
        depth_norm_full = (depth_raw - d_min) / depth_range

    x, y, w, h = face_box.bbox
    x0, y0, x1, y1 = crop_coords
    
    scale_x = size / (x1 - x0)
    scale_y = size / (y1 - y0)
    
    rx0 = max(0, int((x - x0) * scale_x))
    ry0 = max(0, int((y - y0) * scale_y))
    rx1 = min(size, int((x + w - x0) * scale_x))
    ry1 = min(size, int((y + h - y0) * scale_y))
    
    tight_norm = depth_norm_full[ry0:ry1, rx0:rx1]
    
    centre, edge, perimeter_var = _centre_edge_stats(tight_norm)
    var = float(np.var(tight_norm))

    return DepthAnalysis(
        depth_raw=depth_raw,
        depth_normalised_full=depth_norm_full,
        tight_norm=tight_norm,
        tight_box=(rx0, ry0, rx1 - rx0, ry1 - ry0),
        depth_variance=var,
        centre_depth=centre,
        edge_depth=edge,
        centre_minus_edge=float(centre - edge),
        perimeter_variance=perimeter_var # NEW
    )


def classify(analysis: DepthAnalysis, var_threshold: float, centre_threshold: float, perimeter_threshold: float) -> Verdict:
    flag_var = analysis.depth_variance < var_threshold
    flag_centre = analysis.centre_minus_edge < centre_threshold
    flag_perimeter = analysis.perimeter_variance < perimeter_threshold # NEW

    reasons: list[str] = []
    if flag_var:
        reasons.append(f"Depth variance {analysis.depth_variance:.4f} < {var_threshold:.3f} (face region is flat)")
    if flag_centre:
        reasons.append(f"Centre − edge {analysis.centre_minus_edge:+.3f} < {centre_threshold:+.2f} (no nose depth gap)")
    if flag_perimeter:
        reasons.append(f"Perimeter variance {analysis.perimeter_variance:.5f} < {perimeter_threshold:.5f} (border is on a flat plane)")

    return Verdict(
        spoof=bool(reasons),
        reasons=reasons,
        flag_var=flag_var,
        flag_centre=flag_centre,
        flag_perimeter=flag_perimeter
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def depth_to_color_image(depth_norm: np.ndarray) -> np.ndarray:
    img = (np.clip(depth_norm, 0.0, 1.0) * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


def draw_face_box(image_rgb: np.ndarray, bbox: tuple[int, int, int, int], spoof: bool) -> np.ndarray:
    out = image_rgb.copy()
    x, y, w, h = bbox
    color = (220, 60, 60) if spoof else (60, 200, 60)
    thickness = max(2, int(min(w, h) * 0.012))
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render() -> None:
    st.set_page_config(page_title="Layer 5 — Depth PAD", page_icon="📐", layout="wide")
    st.title("Layer 5 · Monocular Depth PAD")
    
    download_with_progress(DEPTH_MODEL_URL, DEPTH_MODEL_PATH, "Depth Anything V2 (~95 MB)")

    with st.sidebar:
        st.header("Thresholds")
        var_thresh = st.number_input("Depth variance threshold", min_value=0.0, max_value=0.5, value=DEFAULT_DEPTH_VAR_THRESHOLD, step=0.005, format="%.3f")
        centre_thresh = st.number_input("Centre − edge depth threshold", min_value=-0.5, max_value=0.5, value=DEFAULT_CENTRE_EDGE_THRESHOLD, step=0.01, format="%.2f")
        perimeter_thresh = st.number_input("Perimeter variance threshold", min_value=0.0, max_value=0.1, value=DEFAULT_PERIMETER_VAR_THRESHOLD, step=0.001, format="%.4f")
        min_face_conf = st.slider("MediaPipe face confidence", min_value=0.1, max_value=0.9, value=DEFAULT_MIN_FACE_CONFIDENCE, step=0.05)

    try:
        session, input_name, det_size = get_depth_session()
    except Exception as exc:
        st.error(f"Failed to load the depth model.\n\n{exc}")
        return

    st.divider()
    source = st.radio("Image source", ("Live camera (st.camera_input)", "Upload file"), horizontal=True)
    
    pil_image: Image.Image | None = None
    if source.startswith("Live"):
        snap = st.camera_input("Take a snapshot")
        if snap is not None:
            pil_image = Image.open(snap)
    else:
        upload = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if upload is not None:
            pil_image = Image.open(upload)

    if pil_image is None:
        st.info("Awaiting input.")
        return

    pil_image = ImageOps.exif_transpose(pil_image)
    rgb_full = np.asarray(pil_image.convert("RGB"))

    h, w = rgb_full.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        rgb_full = cv2.resize(rgb_full, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    face = detect_face(rgb_full, float(min_face_conf))
    if face is None:
        st.error("❌  **No face detected.**")
        st.image(rgb_full, use_container_width=True)
        return

    crop_wide, coords = crop_wide_context(rgb_full, face)

    with st.spinner("Running depth inference…"):
        analysis = analyse(crop_wide, coords, face, session, input_name, det_size)

    verdict = classify(analysis, float(var_thresh), float(centre_thresh), float(perimeter_thresh))

    if verdict.spoof:
        st.error(f"**Verdict · SPOOF SUSPECTED** ({len(verdict.reasons)} flags fired)")
        for r in verdict.reasons:
            st.write(f"  •  {r}")
    else:
        st.success("**Verdict · LIKELY REAL 3-D FACE** (all flags passed)")

    c1, c2, c3, c4 = st.columns(4)
    def metric_card(col, label, value, flagged, hint):
        col.metric(label, value, delta=hint, delta_color="off")
        col.markdown(":red[🚩 flag fired]" if flagged else ":green[✓ valid]")

    metric_card(c1, "Face Depth Variance", f"{analysis.depth_variance:.4f}", verdict.flag_var, f"thresh {var_thresh:.3f}")
    metric_card(c2, "Face Centre − edge", f"{analysis.centre_minus_edge:+.3f}", verdict.flag_centre, f"thresh {centre_thresh:+.2f}")
    metric_card(c3, "Perimeter Variance", f"{analysis.perimeter_variance:.5f}", verdict.flag_perimeter, f"thresh {perimeter_thresh:.4f}")
    c4.metric("Face conf", f"{face.confidence:.2f}", delta="diagnostic", delta_color="off")

    st.divider()
    col_in, col_wide, col_depth = st.columns(3)
    
    col_in.subheader("Input")
    col_in.image(draw_face_box(rgb_full, face.bbox, verdict.spoof), use_container_width=True)
    
    col_wide.subheader("Wide Crop (Context)")
    col_wide.image(draw_face_box(crop_wide, (face.bbox[0]-coords[0], face.bbox[1]-coords[1], face.bbox[2], face.bbox[3]), verdict.spoof), use_container_width=True)
    
    col_depth.subheader("Depth map (w/ Tight Metric Box)")
    colored_depth = depth_to_color_image(analysis.depth_normalised_full)
    tx, ty, tw, th = analysis.tight_box
    cv2.rectangle(colored_depth, (tx, ty), (tx+tw, ty+th), (255, 0, 0), 2)
    col_depth.image(colored_depth, use_container_width=True, caption="Metrics are computed ONLY inside the red box.")

if __name__ == "__main__":
    render()