# 😷 Face Mask Detection

A real-time face mask detection web app built with **Streamlit**, **OpenCV**, and a custom **CNN** trained with TensorFlow/Keras. Detects faces in an uploaded image, webcam snapshot, or live webcam feed, and classifies each as **Mask** or **No Mask** with a confidence score.

<p align="center">
  <img src="assets/banner.png" alt="App banner" width="800"/>
</p>

---

## Features

- **Three input modes** — upload an image, take a webcam snapshot, or run continuous live detection
- **Multi-face detection** — handles multiple faces in a single frame, each labeled independently
- **Robust face detection** — dual Haar cascade fallback + histogram equalization for uneven lighting + adaptive minimum face size (scales to image resolution instead of a fixed pixel size)
- **Tunable detection parameters** — adjust Scale Factor and Min Neighbors live from the sidebar without touching code
- **Clean, card-based UI** — confidence scores and per-face results rendered as readable result cards
- **Cached model + detector loading** — `@st.cache_resource` keeps inference fast across reruns

---

## Demo

| Upload Image | Webcam Snapshot | Live Webcam |
|:---:|:---:|:---:|
| ![Upload demo](assets/demo_upload.png) | ![Snapshot demo](assets/demo_snapshot.png) | ![Live demo](assets/demo_live.png) |

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI / App framework | Streamlit |
| Face detection | OpenCV Haar Cascades |
| Classification model | TensorFlow / Keras CNN |
| Image handling | Pillow, NumPy |

---

## Model

- **Architecture:** 3-block CNN (Conv2D + MaxPooling ×3) → Dense(64) ×2 → Softmax(2)
- **Input:** 128×128 RGB, normalized via a `Rescaling(1./255)` layer built into the model
- **Classes:** `Mask`, `No Mask`
- **Training:** `tf.keras.preprocessing.image_dataset_from_directory` with an 80/20 train/validation split, Adam optimizer, sparse categorical crossentropy, early stopping on validation loss

---

## Project Structure

```
mask-detection-app/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── assets/                 # README screenshots
│   ├── banner.png
│   ├── demo_upload.png
│   ├── demo_snapshot.png
│   └── demo_live.png
└── README.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/itz-jestin/mask-detection-app.git
cd mask-detection-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
streamlit
opencv-python
tensorflow
numpy
pillow
```

### 3. Add your trained model
Place your trained `.h5` model file in the project directory and update the `MODEL_PATH` constant at the top of `app.py`.

### 4. Run the app
```bash
streamlit run app.py
```

---

## Configuration

All key settings live at the top of `app.py`:

```python
MODEL_PATH = "path/to/your_model.h5"
IMG_SIZE = 128
LABELS = ["Mask", "No Mask"]   # must match train_ds.class_names order
```

> **Important:** `LABELS` must match the class order Keras assigned during training. Verify with:
> ```python
> print(train_ds.class_names)
> ```

---

## How It Works

1. Frame is converted to grayscale and histogram-equalized for detection robustness
2. Haar cascade (with an `alt2` fallback) locates faces, using a minimum face size scaled to the image resolution
3. Each detected face is cropped, converted BGR → RGB, resized to 128×128, and passed to the CNN
4. The model's built-in `Rescaling` layer handles normalization — no manual `/255` needed in the app
5. Predicted label and confidence are drawn on the frame and rendered as result cards in the sidebar output

---

## Known Limitations

- Haar cascades work best on clear, front-facing, well-lit faces — profile angles or poor lighting can reduce detection accuracy
- Live Webcam mode requires local execution (browser-based webcam access via `st.camera_input` is used for cloud deployments instead)
- Model accuracy is only as good as the training dataset's diversity in lighting, angles, and mask types

---

## License

MIT

---

## Author

**Jestin**
[GitHub](https://github.com/itz-jestin)<img width="1802" height="763" alt="home" src="https://github.com/user-attachments/assets/eb1593d9-4d70-4c40-8c41-218828853121" />
<img width="1890" height="855" alt="live" src="https://github.com/user-attachments/assets/59f16d41-3717-450d-b1db-48aa27a9b457" />
<img width="1885" height="857" alt="web" src="https://github.com/user-attachments/assets/d79736d8-e55d-43ba-aed3-83afefefcba4" />
<img width="1908" height="831" alt="upload" src="https://github.com/user-attachments/assets/c5398e90-21f0-4c64-9a93-df105bf7aa57" />
