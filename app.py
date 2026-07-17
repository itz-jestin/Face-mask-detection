import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\asus\Downloads\mask_emotion_new.h5"
IMG_SIZE = 128
LABELS = ["Mask", "No Mask"]  # MUST match train_ds.class_names order exactly

st.set_page_config(page_title="Mask Detection", page_icon=":shield:", layout="wide")

# ---------------- STYLE (clean / professional) ----------------
st.markdown("""
<style>
.stApp {
    background: #f5f6f8;
    color: #1a1d23;
}
[data-testid="stSidebar"] {
    background: #1a1d23;
}
[data-testid="stSidebar"] * { color: #e5e7eb !important; }

.app-header {
    padding: 1.25rem 1.5rem;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    margin-bottom: 1.25rem;
}
.app-header h1 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 700;
    color: #111827;
}
.app-header p {
    margin: 0.25rem 0 0 0;
    color: #6b7280;
    font-size: 0.95rem;
}

.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.result-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #9ca3af;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
}
.result-card.mask { border-left-color: #16a34a; }
.result-card.no-mask { border-left-color: #dc2626; }

.result-label { font-weight: 700; font-size: 1.05rem; }
.result-label.mask { color: #16a34a; }
.result-label.no-mask { color: #dc2626; }
.result-conf { color: #6b7280; font-size: 0.9rem; margin-top: 0.15rem; }

.stButton>button {
    border-radius: 6px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL / DETECTORS (cached) ----------------
@st.cache_resource
def load_mask_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_face_cascades():
    # two cascades, tried in order — alt2 catches faces default sometimes misses
    default = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    return default, alt2

model = load_mask_model()
cascade_default, cascade_alt2 = load_face_cascades()

# ---------------- FACE DETECTION ----------------
def detect_faces(frame_bgr, scale_factor, min_neighbors):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # improves detection under uneven/dim lighting

    h, w = gray.shape[:2]
    # scale min face size to the image instead of a fixed 60x60 px,
    # which is too large for small frames and too small for high-res ones
    min_size = (max(20, int(w * 0.08)), max(20, int(h * 0.08)))

    faces = cascade_default.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
    )
    if len(faces) == 0:
        faces = cascade_alt2.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )
    return faces

# ---------------- PREDICTION ----------------
def predict_mask(face_img_bgr):
    # model trained on RGB (image_dataset_from_directory / load_img default) —
    # OpenCV gives BGR, so convert before resizing
    face_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_rgb = face_rgb.astype("float32")  # no /255 — model has Rescaling(1./255) built in
    face_rgb = np.expand_dims(face_rgb, axis=0)
    preds = model.predict(face_rgb, verbose=0)[0]
    idx = int(np.argmax(preds))
    return LABELS[idx], float(preds[idx])

def process_frame(frame_bgr, scale_factor, min_neighbors):
    faces = detect_faces(frame_bgr, scale_factor, min_neighbors)

    results = []
    for (x, y, w, h) in faces:
        face_img = frame_bgr[y:y + h, x:x + w]
        label, conf = predict_mask(face_img)
        results.append((label, conf))

        color = (0, 150, 0) if label == "Mask" else (0, 0, 220)
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {conf*100:.1f}%"
        cv2.putText(frame_bgr, text, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame_bgr, results

def render_results(results):
    if not results:
        st.markdown("""
        <div class="card">
            <b>No face detected.</b>
            <p style="color:#6b7280; margin-top:0.5rem;">
            Try: better lighting, face the camera directly, move closer,
            or lower "Min Neighbors" / decrease "Scale Factor" in the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    for label, conf in results:
        css = "mask" if label == "Mask" else "no-mask"
        st.markdown(f"""
        <div class="result-card {css}">
            <div class="result-label {css}">{label}</div>
            <div class="result-conf">Confidence: {conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("### Settings")
mode = st.sidebar.radio("Input mode", ["Upload Image", "Webcam Snapshot", "Live Webcam"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Detection tuning")
scale_factor = st.sidebar.slider("Scale Factor", 1.03, 1.30, 1.08, 0.01,
                                  help="Lower = more thorough search, slower")
min_neighbors = st.sidebar.slider("Min Neighbors", 2, 8, 4,
                                   help="Lower = more detections, more false positives")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model info")
st.sidebar.caption(f"Input size: {IMG_SIZE}x{IMG_SIZE}")
st.sidebar.caption(f"Classes: {LABELS}")

# ---------------- HEADER ----------------
st.markdown("""
<div class="app-header">
    <h1>Face Mask Detection</h1>
    <p>Upload an image or use your webcam to check for mask compliance in real time.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MODES ----------------
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result_frame, results = process_frame(frame.copy(), scale_factor, min_neighbors)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("**Results**")
            render_results(results)

elif mode == "Webcam Snapshot":
    cam_image = st.camera_input("Take a photo")
    if cam_image is not None:
        image = Image.open(cam_image).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result_frame, results = process_frame(frame.copy(), scale_factor, min_neighbors)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("**Results**")
            render_results(results)

elif mode == "Live Webcam":
    st.markdown('<div class="card">Check the box to start, uncheck to stop.</div>', unsafe_allow_html=True)
    run = st.checkbox("Start Webcam", key="run_check")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        while st.session_state.run_check:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
            frame, _ = process_frame(frame, scale_factor, min_neighbors)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
