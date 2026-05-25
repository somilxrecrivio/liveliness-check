"""
Layer 1: Hardware & Metadata IAD (Injection Attack Detection)
==============================================================

Concept
-------
A genuine camera capturing video has microscopic jitter in its frame delivery
timing. The OS scheduler, sensor readout, USB bus contention, and ISP pipeline
all introduce small random delays between consecutive frames.

A bot or virtual camera that *injects* a pre-rendered video stream typically
emits frames on a perfect, mathematically clean cadence (e.g. exactly 33.33 ms
apart for 30 fps). That perfection is the tell.

We exploit this by computing the variance of inter-frame timestamp deltas:

    delta_i  = t_{i+1} - t_i
    sigma^2  = Var({delta_i})

    sigma^2  ~ 0           --> injection / bot suspected
    sigma^2  > threshold   --> consistent with a real camera

This Streamlit prototype simulates both sources so you can validate the
threshold logic without needing a live camera feed.
"""

import time
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_FPS = 30.0
TARGET_INTERVAL_MS = 1000.0 / TARGET_FPS  # ~33.333 ms per frame at 30 fps

# Variance threshold (ms^2) below which we flag the stream as injected.
# Empirically a real webcam produces deltas with stdev on the order of 1-5 ms,
# i.e. variance in the range ~1-25 ms^2. A perfectly emitted stream sits at 0.
DEFAULT_VARIANCE_THRESHOLD_MS2 = 0.25


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------

def generate_bot_timestamps(n_frames: int, fps: float = TARGET_FPS) -> np.ndarray:
    """
    Simulate a video-injection bot.

    The attacker pipes a pre-rendered file through a virtual camera, so frames
    arrive at the nominal interval with no jitter at all.

    Returns
    -------
    np.ndarray of monotonically increasing frame timestamps in milliseconds.
    """
    interval_ms = 1000.0 / fps
    return np.arange(n_frames, dtype=np.float64) * interval_ms


def generate_real_timestamps(
    n_frames: int,
    fps: float = TARGET_FPS,
    jitter_std_ms: float = 2.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate a real webcam capture.

    Each frame arrives near the nominal interval but with Gaussian jitter that
    models OS scheduling, USB bus contention, and ISP variability.

    Parameters
    ----------
    jitter_std_ms : float
        Standard deviation of the per-frame delay noise. 1-5 ms is realistic
        for a consumer webcam.
    seed : int | None
        Optional seed for reproducible runs.

    Returns
    -------
    np.ndarray of frame timestamps in milliseconds. Guaranteed monotonic.
    """
    rng = np.random.default_rng(seed)
    interval_ms = 1000.0 / fps

    # Per-frame intervals are noisy, but we clamp at a small positive value so
    # the timestamp series remains monotonically increasing.
    noisy_intervals = rng.normal(loc=interval_ms, scale=jitter_std_ms, size=n_frames)
    noisy_intervals = np.clip(noisy_intervals, a_min=0.1, a_max=None)

    timestamps = np.cumsum(noisy_intervals)
    # Anchor the first frame at t = 0 to match the bot simulator's convention.
    timestamps -= timestamps[0]
    return timestamps


# ---------------------------------------------------------------------------
# Live capture (real hardware path)
# ---------------------------------------------------------------------------

def capture_live_timestamps(
    n_frames: int,
    camera_index: int = 0,
    requested_fps: float = TARGET_FPS,
    progress_cb: Callable[[float], None] | None = None,
    preview_slot=None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Open a real camera (or virtual cam) via OpenCV, read `n_frames`, and
    record per-frame timestamps using `time.perf_counter_ns()`.

    Parameters
    ----------
    camera_index : int
        OpenCV device index. 0 is usually the built-in webcam. Virtual cams
        such as OBS Virtual Camera typically appear at index 1+ (varies).
    requested_fps : float
        Hint to the driver — many cameras ignore this and stick to their
        default rate. We don't rely on it for detection, only for matching
        the nominal interval.
    progress_cb : callable, optional
        Receives a fraction in [0, 1] after each frame. Wire this to
        `st.progress` to give the user feedback during long captures.
    preview_slot : streamlit container, optional
        If provided, the most recent frame is shown in this slot so the
        user can see what the camera is actually capturing.

    Returns
    -------
    (timestamps_ms, last_frame_bgr)
        `timestamps_ms` is a monotonic 1-D float64 array anchored at t=0.
        `last_frame_bgr` is the final captured frame (or None if capture
        failed before any frame was read).

    Caveats
    -------
    `time.perf_counter_ns()` is sampled in userspace *after* OpenCV hands us
    the decoded frame, so the variance we measure is the sum of (sensor
    jitter) + (USB transport) + (OS scheduling) + (cv2 decode). That's
    actually *louder* than a hardware-timestamped capture would be, so the
    "real camera" signal should be even easier to distinguish from an
    injected one in this prototype. In production you would prefer
    hardware-level timestamps (V4L2 buffer ts, AVFoundation
    `CMSampleBufferGetPresentationTimeStamp`, WebRTC `captureTimeMs`).
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {camera_index}. "
            "On macOS the first run triggers a permissions prompt — grant "
            "access and try again. To test injection, install OBS Virtual "
            "Camera and pick its index."
        )

    cap.set(cv2.CAP_PROP_FPS, requested_fps)

    timestamps_ns: list[int] = []
    last_frame: np.ndarray | None = None

    try:
        # Discard the first couple of frames — many drivers return stale
        # buffers or take a moment to settle the exposure pipeline, which
        # would inflate the variance of the first delta.
        for _ in range(3):
            cap.read()

        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Frame read failed at index {i}.")
            timestamps_ns.append(time.perf_counter_ns())
            last_frame = frame
            if progress_cb is not None:
                progress_cb((i + 1) / n_frames)
            if preview_slot is not None and (i % 5 == 0):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_slot.image(rgb, channels="RGB", use_container_width=True)
    finally:
        cap.release()

    timestamps_ms = np.asarray(timestamps_ns, dtype=np.float64) / 1e6
    timestamps_ms -= timestamps_ms[0]
    return timestamps_ms, last_frame


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def analyse_timestamps(timestamps: np.ndarray) -> dict:
    """
    Compute the diagnostic statistics on a sequence of frame timestamps.

    Returns
    -------
    dict with keys:
        deltas_ms     : np.ndarray of inter-frame intervals
        mean_ms       : mean inter-frame interval
        std_ms        : standard deviation of the deltas
        variance_ms2  : variance of the deltas (the headline metric)
        min_ms, max_ms: range of deltas
    """
    deltas = np.diff(timestamps)
    return {
        "deltas_ms": deltas,
        "mean_ms": float(np.mean(deltas)),
        "std_ms": float(np.std(deltas)),
        "variance_ms2": float(np.var(deltas)),
        "min_ms": float(np.min(deltas)),
        "max_ms": float(np.max(deltas)),
    }


def classify(variance_ms2: float, threshold_ms2: float) -> tuple[str, str]:
    """
    Apply the variance threshold and return a (verdict, css_color) tuple.

    A variance below the threshold means the timing is "too clean" and is
    flagged as a likely injection attack.
    """
    if variance_ms2 < threshold_ms2:
        return "INJECTION SUSPECTED", "red"
    return "LIKELY REAL CAMERA", "green"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render() -> None:
    st.set_page_config(page_title="Layer 1 — Metadata IAD", page_icon="📹")

    st.title("Layer 1 · Hardware & Metadata IAD")
    st.caption(
        "Detect injected video streams by analysing the variance of "
        "inter-frame timestamp deltas."
    )

    with st.expander("How the detection works", expanded=False):
        st.markdown(
            """
            A real webcam introduces small, unavoidable jitter into frame
            delivery times. A bot injecting a pre-rendered video produces a
            perfect cadence.

            - **Bot:** `delta = 33.33 ms` every time → variance ≈ 0
            - **Real:** `delta ~ N(33.33, σ²)` → variance > 0

            We flag any stream whose delta-variance falls below the
            configurable threshold.

            **Three frame sources** are available:

            1. **Bot (simulated injection)** — synthetic perfect cadence
               for sanity-checking the detector.
            2. **Real camera (simulated jitter)** — synthetic noisy cadence;
               useful for tuning the threshold without hardware.
            3. **Live webcam capture** — opens your real camera via OpenCV
               and records actual frame timestamps. To test injection,
               point this at OBS Virtual Camera replaying a recorded file.
            """
        )

    # ----- Sidebar controls ------------------------------------------------
    with st.sidebar:
        st.header("Controls")

        source = st.radio(
            "Frame source",
            options=(
                "Bot (simulated injection)",
                "Real camera (simulated jitter)",
                "Live webcam capture",
            ),
            help="Pick a synthetic source for math checks or 'Live webcam "
                 "capture' to run against your real camera.",
        )

        n_frames = st.slider(
            "Number of frames",
            min_value=30,
            max_value=600,
            value=150,
            step=10,
            help="How many frames to simulate or capture "
                 "(e.g. 150 ≈ 5 s at 30 fps).",
        )

        camera_index = st.number_input(
            "Camera index (live capture)",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="OpenCV device index. 0 = built-in webcam. Virtual cams "
                 "(OBS) usually appear at 1+.",
        )

        fps = st.slider(
            "Target FPS",
            min_value=15.0,
            max_value=60.0,
            value=TARGET_FPS,
            step=1.0,
        )

        jitter_std_ms = st.slider(
            "Jitter σ (ms) — real-camera only",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Std. dev. of the Gaussian noise injected into real-camera "
                 "intervals. Set to 0 to make the real signal look like a bot.",
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2**31 - 1,
            value=42,
            step=1,
        )

        threshold = st.number_input(
            "Variance threshold (ms²)",
            min_value=0.0,
            value=DEFAULT_VARIANCE_THRESHOLD_MS2,
            step=0.05,
            format="%.2f",
            help="Streams with delta-variance below this value are flagged "
                 "as injected.",
        )

    # ----- Generate or capture timestamps ---------------------------------
    timestamps: np.ndarray | None = None
    captured_frame: np.ndarray | None = None

    if source == "Bot (simulated injection)":
        timestamps = generate_bot_timestamps(n_frames=n_frames, fps=fps)
    elif source == "Real camera (simulated jitter)":
        timestamps = generate_real_timestamps(
            n_frames=n_frames,
            fps=fps,
            jitter_std_ms=jitter_std_ms,
            seed=int(seed),
        )
    else:
        # Live webcam capture — gated behind a button so we don't open the
        # camera on every Streamlit rerun (which would flash the LED and
        # block other UI interactions).
        st.subheader("Live webcam capture")
        st.caption(
            "Click **Capture** to read frames from camera index "
            f"`{int(camera_index)}` and record their arrival times. "
            "Threshold and other settings re-evaluate without re-capturing."
        )

        col_btn, col_preview = st.columns([1, 2])
        with col_btn:
            do_capture = st.button("📷  Capture", type="primary")
            if "live_ts" in st.session_state:
                if st.button("Clear capture"):
                    st.session_state.pop("live_ts", None)
                    st.session_state.pop("live_frame", None)
                    st.rerun()

        preview_slot = col_preview.empty()

        if do_capture:
            progress = st.progress(0.0, text="Opening camera…")
            try:
                ts, last_frame = capture_live_timestamps(
                    n_frames=n_frames,
                    camera_index=int(camera_index),
                    requested_fps=fps,
                    progress_cb=lambda p: progress.progress(
                        p, text=f"Capturing… {int(p * 100)}%"
                    ),
                    preview_slot=preview_slot,
                )
                st.session_state["live_ts"] = ts
                st.session_state["live_frame"] = last_frame
                progress.empty()
                st.success(f"Captured {len(ts)} frames.")
            except RuntimeError as exc:
                progress.empty()
                st.error(str(exc))

        timestamps = st.session_state.get("live_ts")
        captured_frame = st.session_state.get("live_frame")

        if timestamps is None:
            st.info(
                "No capture yet. Pick a camera index and click **Capture**. "
                "To validate the injection-detection logic, point OpenCV at "
                "OBS Virtual Camera (typically index 1+) playing back a "
                "recorded video — variance should collapse toward zero."
            )
            return

        if captured_frame is not None:
            preview_slot.image(
                cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                caption="Last captured frame",
                use_container_width=True,
            )

    stats = analyse_timestamps(timestamps)
    verdict, color = classify(stats["variance_ms2"], threshold)

    # ----- Verdict ---------------------------------------------------------
    st.markdown(
        f"### Verdict · <span style='color:{color}'>{verdict}</span>",
        unsafe_allow_html=True,
    )

    # ----- Metrics ---------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Δ (ms)", f"{stats['mean_ms']:.3f}")
    col2.metric("Std Δ (ms)", f"{stats['std_ms']:.3f}")
    col3.metric("Variance Δ (ms²)", f"{stats['variance_ms2']:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Min Δ (ms)", f"{stats['min_ms']:.3f}")
    col5.metric("Max Δ (ms)", f"{stats['max_ms']:.3f}")
    col6.metric("Threshold (ms²)", f"{threshold:.3f}")

    # ----- Delta time-series chart ----------------------------------------
    st.subheader("Inter-frame deltas over time")
    delta_df = pd.DataFrame(
        {"frame_index": np.arange(len(stats["deltas_ms"])),
         "delta_ms": stats["deltas_ms"]}
    ).set_index("frame_index")
    st.line_chart(delta_df, height=240)

    # ----- Delta histogram -------------------------------------------------
    st.subheader("Distribution of deltas")
    hist_counts, hist_edges = np.histogram(stats["deltas_ms"], bins=30)
    hist_df = pd.DataFrame(
        {"count": hist_counts},
        index=pd.Index(
            (hist_edges[:-1] + hist_edges[1:]) / 2,
            name="delta_ms_bin_center",
        ),
    )
    st.bar_chart(hist_df, height=240)

    # ----- Raw data --------------------------------------------------------
    with st.expander("Raw timestamps & deltas"):
        raw_df = pd.DataFrame(
            {
                "timestamp_ms": timestamps,
                "delta_ms": np.concatenate([[np.nan], stats["deltas_ms"]]),
            }
        )
        st.dataframe(raw_df, use_container_width=True)


if __name__ == "__main__":
    render()
