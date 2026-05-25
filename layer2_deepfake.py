import math
import cv2
import cvzone
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.50 

# YOLO configuration
CLASS_NAMES = ["real", "fake"]  # Swap if YOLO gets the base prediction backwards
MODEL_PATH = "best.pt"          

# Math / Physics configuration
# This threshold determines how much "invisible screen grid" is allowed.
# You will need to tweak this number based on your webcam's resolution!
FFT_VARIANCE_THRESHOLD = 1500  

@st.cache_resource
def load_yolo_model():
    return YOLO(MODEL_PATH)

def calculate_fft_variance(face_crop):
    """
    Mathematical PAD: Converts the face crop into the frequency domain
    to detect the unnatural grid structures of digital screens or paper dots.
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    
    # Perform Discrete Fourier Transform
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Calculate magnitude spectrum
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-7)
    
    # Screens/Prints have high variance in frequency peaks. Real skin is smooth.
    variance = np.var(magnitude_spectrum)
    return variance

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def render():
    st.set_page_config(page_title="Hybrid IAD Layer", layout="wide")
    
    st.markdown("""
        <style>
        .stApp { background-color: #18181b; color: #a1a1aa; }
        .block-container { border: 2px dashed #3f3f46; border-radius: 12px; padding: 2rem; margin-top: 2rem;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Layer 2 · Hybrid Detection (YOLO + Math)")
    st.caption("Fuses Deep Learning bounding boxes with Mathematical Frequency Analysis.")

    try:
        model = load_yolo_model()
    except Exception as e:
        st.error(f"Failed to load YOLO model. Error: {e}")
        return

    source = st.radio("Mode Selection", ("Live Real-Time Feed", "Upload Static Image"), horizontal=True)
    st.divider()

    if source == "Live Real-Time Feed":
        st.info("Live feed started. Check your local webcam.")
        run_camera = st.checkbox("🟢 Start / Stop Camera", value=False)
        
        FRAME_WINDOW = st.empty()
        
        if run_camera:
            cap = cv2.VideoCapture(0) 
            cap.set(3, 640)
            cap.set(4, 480)
            
            while run_camera:
                success, frame = cap.read()
                if not success:
                    break

                results = model(frame, stream=True, verbose=False)
                
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                        w, h = x2 - x1, y2 - y1
                        
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        
                        if conf > CONFIDENCE_THRESHOLD:
                            # 1. Get YOLO's guess
                            yolo_guess = CLASS_NAMES[cls]
                            
                            # 2. Extract the face and run the Math!
                            # Ensure coordinates are within bounds to prevent crash
                            ih, iw, _ = frame.shape
                            crop_x1, crop_y1 = max(0, x1), max(0, y1)
                            crop_x2, crop_y2 = min(iw, x2), min(ih, y2)
                            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            
                            fft_variance = 0
                            math_guess = "real"
                            
                            if face_crop.size > 0:
                                fft_variance = calculate_fft_variance(face_crop)
                                if fft_variance > FFT_VARIANCE_THRESHOLD:
                                    math_guess = "fake"

                            # 3. The Ensemble Voting System (Fail-Safe)
                            # If either system says fake, it is fake.
                            if yolo_guess == "fake" or math_guess == "fake":
                                final_label = "FAKE"
                                color = (0, 0, 255) # Red
                            else:
                                final_label = "REAL"
                                color = (0, 255, 0) # Green
                            
                            # Draw UI
                            cvzone.cornerRect(frame, (x1, y1, w, h), colorC=color, colorR=color, l=20, t=3)
                            
                            # Display diagnostic text to show what both systems are thinking
                            diagnostic_text = f"{final_label} (YOLO:{yolo_guess[0].upper()} | FFT:{int(fft_variance)})"
                            cvzone.putTextRect(frame, diagnostic_text, 
                                               (max(0, x1), max(35, y1 - 10)), scale=1.2, thickness=2,
                                               colorR=color, colorB=color)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb, use_container_width=True)
            
            cap.release()

    else:
        upload = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
        if upload:
            pil_image = Image.open(upload)
            pil_image = ImageOps.exif_transpose(pil_image)
            frame = np.array(pil_image.convert("RGB"))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = model(frame_bgr, stream=True, verbose=False)
            
            detected = False
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                    w, h = x2 - x1, y2 - y1
                    
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])

                    if conf > CONFIDENCE_THRESHOLD:
                        detected = True
                        yolo_guess = CLASS_NAMES[cls]
                        
                        ih, iw, _ = frame_bgr.shape
                        crop_x1, crop_y1 = max(0, x1), max(0, y1)
                        crop_x2, crop_y2 = min(iw, x2), min(ih, y2)
                        face_crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        fft_variance = 0
                        math_guess = "real"
                        
                        if face_crop.size > 0:
                            fft_variance = calculate_fft_variance(face_crop)
                            if fft_variance > FFT_VARIANCE_THRESHOLD:
                                math_guess = "fake"

                        if yolo_guess == "fake" or math_guess == "fake":
                            final_label = "FAKE"
                            color = (0, 0, 255)
                        else:
                            final_label = "REAL"
                            color = (0, 255, 0)
                        
                        cvzone.cornerRect(frame_bgr, (x1, y1, w, h), colorC=color, colorR=color, l=30, t=4)
                        diagnostic_text = f"{final_label} (YOLO:{yolo_guess[0].upper()} | FFT:{int(fft_variance)})"
                        cvzone.putTextRect(frame_bgr, diagnostic_text, 
                                           (max(0, x1), max(35, y1 - 10)), scale=1.5, thickness=2,
                                           colorR=color, colorB=color)

            output_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(output_rgb, caption="Hybrid Detection Output", use_container_width=True)
            
            if not detected:
                st.warning("No faces detected above the confidence threshold.")

if __name__ == "__main__":
    render()