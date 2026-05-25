"""
Layer 5: Monocular Depth PAD
=============================

Concept
-------
A real face has visible 3-D structure: the nose protrudes 2–4 cm
beyond the cheeks, which are themselves closer than the ears /
background. A 2-D presentation attack (printed photo, displayed
screen) doesn't — the depicted face is rendered on a flat surface
and every pixel sits at the same physical depth.

Method
------
1.  MediaPipe BlazeFace → face bbox.
2.  Crop the face with a 15 % margin so the corners of the crop
    include cheek / hair / background.
3.  Run **Depth Anything V2 Small** (~95 MB ONNX, auto-downloaded
    to ./models/) on the face crop. Outputs an inverse-depth map at
    the model's native input resolution.
4.  Normalise the depth map to `[0, 1]` to be invariant to the
    model's scene-scale guess.
5.  Compute two PAD flags:

      - **Depth variance** of the normalised map. A real face has
        ≥ 0.03 variance from the nose-to-cheek-to-corner spread.
        A flat photo collapses below this.
      - **Centre minus edge depth.** Mean over the central 30 % vs
        mean over the four corner 15 % patches. A real face has the
        centre measurably closer than the corners; a 2-D photo is
        ≈ 0.

6.  ANY flag fires → SPOOF.

Run:
    streamlit run layer5_depth_pad.py

First run downloads:
- BlazeFace short-range (~230 KB)
- Depth Anything V2 Small ONNX (~95 MB)
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

# Depth Anything V2 Small — HuggingFace mirror with stable filename.
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

# Depth Anything V2 Small standard input resolution
DEPTH_INPUT_SIZE = 518
# ImageNet normalisation matches the original Depth Anything training recipe.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_DEPTH_VAR_THRESHOLD = 0.030
DEFAULT_CENTRE_EDGE_THRESHOLD = 0.10
DEFAULT_MIN_FACE_CONFIDENCE = 0.5
FACE_CROP_MARGIN = 0.15

# MediaPipe Tasks API attribute aliases (see Layer 3 for rationale)
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions


@dataclass
class FaceBox:
    bbox: tuple[int, int, int, int]   # x, y, w, h
    confidence: float


@dataclass
class DepthAnalysis:
    depth_raw: np.ndarray             # 2-D, model output, fp32
    depth_normalised: np.ndarray      # 2-D, [0, 1]
    depth_variance: float
    depth_range: float                # raw max - min
    centre_depth: float               # normalised, central 30 %
    edge_depth: float                 # normalised, four corner 15 %
    centre_minus_edge: float          # the headline 3-D-structure metric


@dataclass
class Verdict:
    spoof: bool
    reasons: list
    flag_var: bool
    flag_centre: bool


# ---------------------------------------------------------------------------
# Lazy model loading + auto-download
# ---------------------------------------------------------------------------

_FACE_DETECTOR = None
_FACE_DETECTOR_CONF: float | None = None


def _ensure_download(url: str, path: str) -> str:
    """Download a model to `path` if not already present."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _get_face_detector(min_confidence: float):
    """Cached MediaPipe FaceDetector; rebuilt when confidence changes."""
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


@st.cache_resource(show_spinner=False)
def get_depth_session():
    """
    Cached ONNX Runtime session for Depth Anything V2 Small.

    Resolves the input tensor name and the spatial input size from the
    model itself; falls back to DEPTH_INPUT_SIZE if the dim is dynamic.
    """
    path = _ensure_download(DEPTH_MODEL_URL, DEPTH_MODEL_PATH)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    name = inp.name
    shape = inp.shape  # e.g. [1, 3, 518, 518] or [1, 3, "height", "width"]
    size = DEPTH_INPUT_SIZE
    if len(shape) >= 4 and isinstance(shape[2], int) and shape[2] > 0:
        size = int(shape[2])
    return session, name, size


# ---------------------------------------------------------------------------
# Face detection + crop
# ---------------------------------------------------------------------------

def detect_face(image_rgb: np.ndarray, min_confidence: float) -> FaceBox | None:
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
    conf = float(det.categories[0].score) if det.categories else 0.0
    return FaceBox(bbox=(x, y, w, h), confidence=conf)


def crop_face(
    image_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: float = FACE_CROP_MARGIN,
) -> np.ndarray:
    x, y, w, h = bbox
    h_img, w_img = image_rgb.shape[:2]
    m = int(margin * min(w, h))
    x0 = max(0, x - m)
    y0 = max(0, y - m)
    x1 = min(w_img, x + w + m)
    y1 = min(h_img, y + h + m)
    return image_rgb[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Depth inference + analysis
# ---------------------------------------------------------------------------

def preprocess_for_depth(face_rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize to (size, size), ImageNet-normalise, CHW, add batch."""
    resized = cv2.resize(face_rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[None]
    return arr.astype(np.float32)


def infer_depth(
    face_rgb: np.ndarray,
    session: ort.InferenceSession,
    input_name: str,
    size: int,
) -> np.ndarray:
    tensor = preprocess_for_depth(face_rgb, size)
    outputs = session.run(None, {input_name: tensor})
    depth = outputs[0]
    # Output is typically [1, H, W] or [1, 1, H, W]; squeeze down to 2-D.
    while depth.ndim > 2:
        depth = depth[0]
    return depth.astype(np.float32)


def _centre_edge_stats(
    depth_norm: np.ndarray,
    centre_frac: float = 0.30,
    edge_frac: float = 0.15,
) -> tuple[float, float]:
    """Mean depth in the central patch and across the four corner patches."""
    h, w = depth_norm.shape
    cy, cx = h // 2, w // 2
    half = int(centre_frac * min(h, w) / 2)
    es = max(1, int(edge_frac * min(h, w)))

    centre_patch = depth_norm[
        max(0, cy - half) : cy + half,
        max(0, cx - half) : cx + half,
    ]
    centre = float(centre_patch.mean()) if centre_patch.size else 0.0

    corners = [
        depth_norm[:es, :es].mean(),
        depth_norm[:es, w - es :].mean(),
        depth_norm[h - es :, :es].mean(),
        depth_norm[h - es :, w - es :].mean(),
    ]
    edge = float(np.mean(corners))
    return centre, edge


def analyse(
    face_rgb: np.ndarray,
    session: ort.InferenceSession,
    input_name: str,
    size: int,
) -> DepthAnalysis:
    depth_raw = infer_depth(face_rgb, session, input_name, size)
    d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
    depth_range = d_max - d_min
    if depth_range < 1e-12:
        depth_norm = np.zeros_like(depth_raw)
    else:
        depth_norm = (depth_raw - d_min) / depth_range

    centre, edge = _centre_edge_stats(depth_norm)
    return DepthAnalysis(
        depth_raw=depth_raw,
        depth_normalised=depth_norm,
        depth_variance=float(np.var(depth_norm)),
        depth_range=depth_range,
        centre_depth=centre,
        edge_depth=edge,
        centre_minus_edge=float(centre - edge),
    )


def classify(
    analysis: DepthAnalysis,
    var_threshold: float = DEFAULT_DEPTH_VAR_THRESHOLD,
    centre_threshold: float = DEFAULT_CENTRE_EDGE_THRESHOLD,
) -> Verdict:
    flag_var = analysis.depth_variance < var_threshold
    flag_centre = analysis.centre_minus_edge < centre_threshold

    reasons: list[str] = []
    if flag_var:
        reasons.append(
            f"Depth variance {analysis.depth_variance:.4f} < "
            f"threshold {var_threshold:.4f} (face crop too flat — 2-D attack)"
        )
    if flag_centre:
        reasons.append(
            f"Centre − edge depth {analysis.centre_minus_edge:+.3f} < "
            f"threshold {centre_threshold:+.2f} (no nose-vs-corner depth gap)"
        )

    return Verdict(
        spoof=bool(reasons),
        reasons=reasons,
        flag_var=flag_var,
        flag_centre=flag_centre,
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def depth_to_color_image(depth_norm: np.ndarray) -> np.ndarray:
    """VIRIDIS-colormap a normalised depth map for display."""
    img = (np.clip(depth_norm, 0.0, 1.0) * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


def draw_face_box(
    image_rgb: np.ndarray, bbox: tuple[int, int, int, int], spoof: bool
) -> np.ndarray:
    out = image_rgb.copy()
    x, y, w, h = bbox
    color = (220, 60, 60) if spoof else (60, 200, 60)
    thickness = max(2, int(min(w, h) * 0.012))
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    label = "SPOOF" if spoof else "REAL"
    font_scale = max(0.5, min(w, h) * 0.004)
    cv2.putText(
        out, label, (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
        max(1, int(thickness * 0.6)), cv2.LINE_AA,
    )
    return out


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render() -> None:
    st.set_page_config(
        page_title="Layer 5 — Depth PAD",
        page_icon="📐",
        layout="wide",
    )
    st.title("Layer 5 · Monocular Depth PAD")
    st.caption(
        "Detect 2-D presentation attacks (printed photo, displayed screen) "
        "by checking whether the face crop has real 3-D depth structure."
    )

    with st.expander("How it works", expanded=False):
        st.markdown(
            f"""
            A real face has visible 3-D structure: the nose protrudes
            beyond the cheeks. A 2-D photo is flat — every pixel sits at
            the same physical depth.

            We run **Depth Anything V2 Small** ({DEPTH_INPUT_SIZE}×
            {DEPTH_INPUT_SIZE} input, ~95 MB ONNX, CPU) on a face crop
            and compute two flags:

            1. **Depth variance** of the normalised map. Real faces
               produce ~0.03+ variance across the face ROI from
               nose-to-cheek-to-corner spread. Flat photos collapse
               below.
            2. **Centre minus edge depth.** Mean over the central 30 %
               vs four corner 15 % patches. Real faces have the centre
               (nose) measurably closer than the corners (cheeks /
               background). 2-D photos are ≈ 0.

            ANY flag fires → SPOOF.

            First run downloads:
            - BlazeFace short-range face detector (~230 KB)
            - Depth Anything V2 Small ONNX (~95 MB) to `./models/`
            """
        )

    with st.sidebar:
        st.header("Thresholds")
        var_thresh = st.number_input(
            "Depth variance threshold",
            min_value=0.0,
            max_value=0.5,
            value=DEFAULT_DEPTH_VAR_THRESHOLD,
            step=0.005,
            format="%.3f",
            help="Below this is flagged as a 2-D attack. Real faces ~0.04+.",
        )
        centre_thresh = st.number_input(
            "Centre − edge depth threshold",
            min_value=-0.5,
            max_value=0.5,
            value=DEFAULT_CENTRE_EDGE_THRESHOLD,
            step=0.01,
            format="%.2f",
            help="Below this means no nose-vs-corner depth gap → 2-D attack.",
        )
        min_face_conf = st.slider(
            "MediaPipe face confidence",
            min_value=0.1,
            max_value=0.9,
            value=DEFAULT_MIN_FACE_CONFIDENCE,
            step=0.05,
        )

    with st.spinner(
        "Loading Depth Anything V2 Small (first run downloads ~95 MB)…"
    ):
        try:
            session, input_name, det_size = get_depth_session()
        except Exception as exc:
            st.error(
                f"Failed to load the depth model.\n\n{exc}\n\n"
                f"You can manually download from:\n```\n{DEPTH_MODEL_URL}\n"
                f"```\nand place it at: `{DEPTH_MODEL_PATH}`"
            )
            return

    st.divider()
    source = st.radio(
        "Image source",
        ("Live camera (st.camera_input)", "Upload file"),
        horizontal=True,
    )
    pil_image: Image.Image | None = None
    if source.startswith("Live"):
        snap = st.camera_input("Take a snapshot")
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
        st.info(
            """
            **Try these test cases:**
            1. **Real face from your webcam** → expect REAL.
            2. **Photo of a printed face held to the webcam** →
               expect SPOOF (variance + centre flags).
            3. **Photo of your phone screen showing a face** →
               expect SPOOF.
            4. **Photo of a wall, no face** → no verdict (face required).
            """
        )
        return

    pil_image = ImageOps.exif_transpose(pil_image)
    rgb_full = np.asarray(pil_image.convert("RGB"))

    # Resize huge images for inference speed; depth quality is preserved at
    # 1024 px since the model resamples to its 518 native size anyway.
    h, w = rgb_full.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        rgb_full = cv2.resize(
            rgb_full,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    face = detect_face(rgb_full, float(min_face_conf))
    if face is None:
        st.error(
            "❌  **No face detected.** Layer 5 metrics are only meaningful "
            "on a face ROI — re-shoot with a clearer, front-on face."
        )
        st.image(rgb_full, use_container_width=True)
        return

    face_crop = crop_face(rgb_full, face.bbox)

    with st.spinner("Running depth inference…"):
        analysis = analyse(face_crop, session, input_name, det_size)

    verdict = classify(analysis, float(var_thresh), float(centre_thresh))

    # ----- Verdict banner -----
    if verdict.spoof:
        st.error(
            f"**Verdict · SPOOF SUSPECTED**  "
            f"({len(verdict.reasons)} flag"
            f"{'s' if len(verdict.reasons) > 1 else ''} fired)"
        )
        for r in verdict.reasons:
            st.write(f"  •  {r}")
    else:
        st.success(
            "**Verdict · LIKELY REAL 3-D FACE**  (both flags passed)"
        )

    # ----- Metrics -----
    c1, c2, c3, c4 = st.columns(4)

    def metric_card(col, label, value, flagged, hint):
        col.metric(label, value, delta=hint, delta_color="off")
        col.markdown(
            ":red[🚩 flag fired]" if flagged else ":green[✓ within natural range]"
        )

    metric_card(
        c1,
        "Depth variance",
        f"{analysis.depth_variance:.4f}",
        verdict.flag_var,
        f"thresh {var_thresh:.3f}",
    )
    metric_card(
        c2,
        "Centre − edge depth",
        f"{analysis.centre_minus_edge:+.3f}",
        verdict.flag_centre,
        f"thresh {centre_thresh:+.2f}",
    )
    c3.metric(
        "Raw depth range",
        f"{analysis.depth_range:.2f}",
        delta="diagnostic only",
        delta_color="off",
    )
    c4.metric(
        "Face confidence",
        f"{face.confidence:.2f}",
        delta="diagnostic only",
        delta_color="off",
    )

    # ----- Visualisations -----
    st.divider()
    col_in, col_face, col_depth = st.columns(3)
    col_in.subheader("Input + face bbox")
    col_in.image(
        draw_face_box(rgb_full, face.bbox, verdict.spoof),
        use_container_width=True,
    )
    col_face.subheader("Face crop (analysed)")
    col_face.image(face_crop, use_container_width=True)
    col_depth.subheader("Depth map")
    col_depth.image(
        depth_to_color_image(analysis.depth_normalised),
        use_container_width=True,
        caption="VIRIDIS colormap. Yellow = closer; purple = farther.",
    )

    # ----- Diagnostics -----
    with st.expander("Depth diagnostics"):
        st.write(
            f"**Centre depth** (mean of central 30 %): "
            f"`{analysis.centre_depth:.3f}`"
        )
        st.write(
            f"**Edge depth** (mean of four corner 15 %): "
            f"`{analysis.edge_depth:.3f}`"
        )
        st.write(
            f"**Centre − Edge** (the 3-D-structure metric): "
            f"`{analysis.centre_minus_edge:+.3f}`"
        )
        st.write(
            f"**Raw depth range** (max − min before normalisation): "
            f"`{analysis.depth_range:.4f}`"
        )
        st.write(
            f"**Normalised variance**: `{analysis.depth_variance:.4f}`"
        )


if __name__ == "__main__":
    render()
