"""
Layer 4: Biometric Matching
============================

Concept
-------
Once the upstream layers have proved the object in front of the camera
is a live, physical human face, this layer answers the question:
*is it the same person as the stored reference ID?*

We use the **InsightFace** ArcFace-trained recognition model
(`buffalo_l` bundle, W600K-R50 backbone) to extract a 512-dimensional
L2-normalised embedding per face. The embedding lives on the unit
hypersphere and is trained so that:

    same identity      → cosine similarity  ≈  0.50 – 1.00
    different identity → cosine similarity  ≈ −0.20 – 0.30

Cosine similarity for two L2-normalised vectors is just their dot
product:

    sim = embedding_ref · embedding_probe   ∈ [−1, +1]

We declare a match when `sim ≥ threshold`. The default 0.40 is the
standard balanced operating point on the LFW / CFP-FP / AgeDB
benchmarks: well above the false-accept rate of unrelated faces,
well below the false-reject rate of legitimate self-pairs across
variation in age, lighting, and pose.

Run:
    streamlit run layer4_biometrics.py

The first run downloads the buffalo_l bundle (~280 MB) to
`~/.insightface/models/`. Subsequent runs load from cache.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "buffalo_l"
DEFAULT_THRESHOLD = 0.40
DEFAULT_DET_SIZE = 640


@dataclass
class FaceData:
    """Output of InsightFace for one detection."""
    bbox: tuple[int, int, int, int]   # x, y, w, h
    kps: np.ndarray                   # 5 keypoints, (5, 2) float
    embedding: np.ndarray             # 512-D, already L2-normalised
    det_score: float                  # detector confidence


@dataclass
class MatchResult:
    similarity: float                 # cosine in [-1, 1]
    distance: float                   # 1 - similarity, for forensic display
    threshold: float
    match: bool


# ---------------------------------------------------------------------------
# Model loading + inference
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(name: str = DEFAULT_MODEL, det_size: int = DEFAULT_DET_SIZE):
    """
    Build and cache the InsightFace FaceAnalysis pipeline.

    `providers=['CPUExecutionProvider']` pins ONNX Runtime to CPU; this
    keeps behaviour predictable across machines and side-steps GPU
    fingerprinting issues on macOS. For production with GPUs available,
    prepend 'CUDAExecutionProvider' to the list.

    `ctx_id=0` is passed because the underlying session honours it for
    backend selection; the explicit providers list above is what
    actually controls the device choice.
    """
    app = FaceAnalysis(name=name, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def _load_pil(image_input) -> np.ndarray:
    """Load an arbitrary file-like into an RGB numpy array, EXIF-corrected."""
    img = Image.open(image_input)
    img = ImageOps.exif_transpose(img)
    return np.asarray(img.convert("RGB"))


def detect_and_embed(model, image_rgb: np.ndarray) -> FaceData | None:
    """
    Run face detection + alignment + embedding on a single RGB image.

    Returns the *largest* face only. InsightFace expects BGR input, so
    we convert at the boundary.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    faces = model.get(image_bgr)
    if not faces:
        return None

    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    f = faces[0]

    x1, y1, x2, y2 = (int(v) for v in f.bbox)
    h_img, w_img = image_rgb.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    return FaceData(
        bbox=(x1, y1, x2 - x1, y2 - y1),
        kps=np.asarray(f.kps, dtype=np.float32),
        embedding=np.asarray(f.normed_embedding, dtype=np.float32),
        det_score=float(f.det_score),
    )


def detect_and_embed_auto_rotate(
    model,
    image_rgb: np.ndarray,
    confident_score: float = 0.6,
) -> tuple[FaceData, np.ndarray, int] | None:
    """
    Try detection at 0°, 90°, 180°, 270° and return the orientation
    that gave the best face. Returns (face, rotated_image, degrees).

    Handles ID uploads (Aadhar, passport scans) shot in portrait when
    the face on the card is meant to be viewed landscape. Short-circuits
    when the original orientation already gives a confident detection
    so the common case stays single-inference.
    """
    face = detect_and_embed(model, image_rgb)
    if face is not None and face.det_score >= confident_score:
        return face, image_rgb, 0

    best_face = face
    best_image = image_rgb
    best_deg = 0
    best_score = face.det_score if face is not None else -1.0

    # np.rot90 with k=1 rotates 90° CCW; we try 90, 180, 270.
    for k in (1, 2, 3):
        rotated = np.ascontiguousarray(np.rot90(image_rgb, k=k))
        candidate = detect_and_embed(model, rotated)
        if candidate is not None and candidate.det_score > best_score:
            best_face = candidate
            best_image = rotated
            best_deg = 90 * k
            best_score = candidate.det_score

    if best_face is None:
        return None
    return best_face, best_image, best_deg


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Dot product — both vectors are already unit-norm from InsightFace."""
    return float(np.dot(emb1, emb2))


def classify(similarity: float, threshold: float) -> MatchResult:
    return MatchResult(
        similarity=similarity,
        distance=1.0 - similarity,
        threshold=threshold,
        match=similarity >= threshold,
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_face_overlay(
    image_rgb: np.ndarray,
    face: FaceData,
    match: bool,
    label: str | None = None,
) -> np.ndarray:
    """Bounding box + 5 keypoints + verdict label."""
    out = image_rgb.copy()
    x, y, w, h = face.bbox
    color = (60, 200, 60) if match else (220, 60, 60)
    thickness = max(2, int(min(w, h) * 0.012))
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)

    for kx, ky in face.kps.astype(int):
        cv2.circle(out, (int(kx), int(ky)), max(2, thickness), color, -1)

    if label:
        font_scale = max(0.5, min(w, h) * 0.004)
        cv2.putText(
            out,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, int(thickness * 0.6)),
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render() -> None:
    st.set_page_config(
        page_title="Layer 4 — Biometric Matching",
        page_icon="🆔",
        layout="wide",
    )
    st.title("Layer 4 · Biometric Matching")
    st.caption(
        "Verify a live probe face matches a stored reference ID via "
        "InsightFace ArcFace embeddings + cosine similarity."
    )

    with st.expander("How the matching works", expanded=False):
        st.markdown(
            """
            **Pipeline.** Each image is fed to InsightFace's pipeline:

            1. **RetinaFace** detector locates faces and 5 keypoints.
            2. **Face alignment** warps the largest face to a canonical
               112×112 crop using the keypoints.
            3. **ArcFace** (W600K-R50 backbone) maps the aligned crop to
               a **512-D unit vector**.

            Both embeddings live on the unit hypersphere by
            construction, so the **cosine similarity** is just their
            dot product:

            ```
            sim = embedding_ref · embedding_probe   ∈ [−1, +1]
            ```

            **Threshold guidance** (W600K-R50 on standard benchmarks):

            | threshold | character           | typical use |
            |-----------|---------------------|-------------|
            | 0.30      | very loose          | risk-tolerant onboarding |
            | **0.40**  | **balanced**        | **default — KYC, login** |
            | 0.50      | strict              | high-stakes auth |
            | 0.60      | very strict         | duplicate detection |

            The strong upstream layers in this stack (1: injection, 2:
            deepfake, 3: spatial PAD) handle the spoof / liveness
            question. Layer 4 only handles *identity*, assuming what
            it's seeing is a live human.
            """
        )

    # ----- Sidebar -----
    with st.sidebar:
        st.header("Match threshold")
        threshold = st.slider(
            "Cosine similarity",
            min_value=0.20,
            max_value=0.80,
            value=DEFAULT_THRESHOLD,
            step=0.01,
            help="Below this → NO MATCH. Tighter threshold = fewer "
                 "false accepts and more false rejects.",
        )
        st.caption(
            f"At threshold {threshold:.2f}, ArcFace W600K-R50 typically "
            "achieves ~0.1–1 % false-accept rate. Tune against a "
            "labelled pair dataset for production."
        )

        st.divider()
        st.subheader("Model")
        det_size = st.slider(
            "Detection size",
            min_value=320,
            max_value=1280,
            value=DEFAULT_DET_SIZE,
            step=64,
            help="Internal detector input size. Higher = finds smaller "
                 "faces, slower. Default 640 is the standard.",
        )
        st.code(f"{DEFAULT_MODEL}  (~280 MB)\nRetinaFace + ArcFace W600K-R50",
                language=None)
        st.caption(
            "Downloads to `~/.insightface/models/` on first use. "
            "Subsequent runs load from cache in ~2 s."
        )

    # ----- Model load -----
    with st.spinner(
        "Loading InsightFace model — first run downloads ~280 MB, "
        "subsequent runs load from cache..."
    ):
        try:
            model = load_model(DEFAULT_MODEL, int(det_size))
        except Exception as exc:
            st.error(f"Failed to load InsightFace model.\n\n{exc}")
            return

    # ----- Inputs -----
    col_ref, col_probe = st.columns(2)
    ref_image: np.ndarray | None = None
    probe_image: np.ndarray | None = None

    with col_ref:
        st.subheader("Reference ID")
        ref_upload = st.file_uploader(
            "Upload reference ID image (passport photo, profile pic, …)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="ref_upload",
        )
        if ref_upload is not None:
            ref_image = _load_pil(ref_upload)

    with col_probe:
        st.subheader("Probe face")
        probe_source = st.radio(
            "Source",
            ("Webcam snapshot", "Upload image"),
            horizontal=True,
            key="probe_source",
        )
        if probe_source == "Webcam snapshot":
            snap = st.camera_input("Take a snapshot", key="probe_cam")
            if snap is not None:
                probe_image = _load_pil(snap)
        else:
            probe_upload = st.file_uploader(
                "Upload probe image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="probe_upload",
            )
            if probe_upload is not None:
                probe_image = _load_pil(probe_upload)

    if ref_image is None or probe_image is None:
        st.info(
            "Provide **both** a reference ID image and a probe face "
            "to compare. The reference is typically a passport / ID "
            "scan; the probe is a fresh capture."
        )
        return

    # ----- Detect + embed (with auto-rotation for portrait ID uploads) -----
    with st.spinner("Running detection + embedding…"):
        ref_result = detect_and_embed_auto_rotate(model, ref_image)
        probe_result = detect_and_embed_auto_rotate(model, probe_image)

    ref_face = probe_face = None
    ref_rotation = probe_rotation = 0
    if ref_result is not None:
        ref_face, ref_image, ref_rotation = ref_result
    if probe_result is not None:
        probe_face, probe_image, probe_rotation = probe_result

    if ref_rotation:
        st.info(f"↻ Reference image auto-rotated {ref_rotation}° to detect face.")
    if probe_rotation:
        st.info(f"↻ Probe image auto-rotated {probe_rotation}° to detect face.")

    if ref_face is None and probe_face is None:
        st.error(
            "❌  No face detected in **either** image. "
            "Re-shoot with clearer, front-on faces."
        )
        return
    if ref_face is None:
        st.error(
            "❌  No face detected in the **reference ID** image. "
            "Upload a clearer ID photo."
        )
        st.image(ref_image, caption="Reference image — no face", use_container_width=True)
        return
    if probe_face is None:
        st.error(
            "❌  No face detected in the **probe** image. "
            "Re-shoot with a clearer, front-on snapshot."
        )
        st.image(probe_image, caption="Probe image — no face", use_container_width=True)
        return

    sim = cosine_similarity(ref_face.embedding, probe_face.embedding)
    result = classify(sim, float(threshold))

    # ----- Verdict banner -----
    if result.match:
        st.success(
            f"### ✓ MATCH  ·  cosine similarity {sim:.4f} ≥ "
            f"threshold {threshold:.2f}"
        )
    else:
        st.error(
            f"### ✗ NO MATCH  ·  cosine similarity {sim:.4f} < "
            f"threshold {threshold:.2f}"
        )

    # ----- Metric cards -----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Cosine similarity",
        f"{sim:.4f}",
        delta=f"thresh {threshold:.2f}",
        delta_color="off",
    )
    c2.metric("Cosine distance", f"{result.distance:.4f}")
    c3.metric(
        "Ref det. confidence",
        f"{ref_face.det_score:.3f}",
        delta="0.5+ ideal",
        delta_color="off",
    )
    c4.metric(
        "Probe det. confidence",
        f"{probe_face.det_score:.3f}",
        delta="0.5+ ideal",
        delta_color="off",
    )

    # ----- Annotated images -----
    st.divider()
    col_ref_v, col_probe_v = st.columns(2)
    label = "MATCH" if result.match else "NO MATCH"

    with col_ref_v:
        st.subheader("Reference + detected face")
        annotated_ref = draw_face_overlay(
            ref_image, ref_face, result.match, label=label
        )
        st.image(
            annotated_ref,
            use_container_width=True,
            caption=f"Detection score: {ref_face.det_score:.3f}",
        )

    with col_probe_v:
        st.subheader("Probe + detected face")
        annotated_probe = draw_face_overlay(
            probe_image, probe_face, result.match, label=label
        )
        st.image(
            annotated_probe,
            use_container_width=True,
            caption=f"Detection score: {probe_face.det_score:.3f}",
        )

    # ----- Embedding diagnostics -----
    with st.expander("Embedding diagnostics", expanded=False):
        st.markdown(
            "Both embeddings are 512-D, L2-normalised. Showing the **first "
            "32 dimensions** for inspection."
        )
        diag_df = pd.DataFrame(
            {
                "reference": ref_face.embedding[:32],
                "probe": probe_face.embedding[:32],
            },
            index=pd.Index(range(32), name="dim"),
        )
        st.line_chart(diag_df, height=240)

        st.markdown("**Per-dimension element-wise product** (first 32 dims)")
        prod = ref_face.embedding * probe_face.embedding
        prod_df = pd.DataFrame(
            {"ref ⊙ probe": prod[:32]},
            index=pd.Index(range(32), name="dim"),
        )
        st.bar_chart(prod_df, height=180)
        st.caption(
            f"Sum across **all 512 dims** = cosine similarity = "
            f"**{sim:.4f}**. The bars show how each individual dimension "
            "contributes; large positive bars mean aligned features, "
            "negative means anti-aligned."
        )

        positive = float((prod > 0).sum())
        st.write(
            f"Dimensions with positive contribution: "
            f"**{int(positive)} / 512** "
            f"({positive / 512 * 100:.1f}%)."
        )


if __name__ == "__main__":
    render()
