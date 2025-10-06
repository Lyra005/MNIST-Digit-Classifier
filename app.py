import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np

st.set_page_config(page_title="Handwritten Digit Recognizer ", layout="centered")

st.markdown(
    """
    <style>
    
    /*Toolbar*/
    .st-emotion-cache-gquqoo {
        background:#83c5be;
    }
    
    .block-container {
        display: flex;
        flex-direction: column;
        align-items: center;   /* horizontal center */
        justify-content: center; /* vertical center (with full height) */
        text-align: center;
    }
    
     </style>
    """,
    unsafe_allow_html=True
)

# Load Keras model
model = tf.keras.models.load_model("model.keras", compile=False)

# UI
st.title("🖊️ Handwritten Digit Recognizer (MNIST)")
st.write("Draw a digit or upload an image. Let the AI model tell you what it sees!")

def _preprocess_core(gray_img):
    # Threshold to isolate the digit
    _, img_bin = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find external contours
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    # Largest contour assumed to be the digit
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    digit = img_bin[y:y + h, x:x + w]

    # Resize keeping aspect ratio to 20x20
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # Pad to 28x28
    padded = np.pad(digit, ((4, 4), (4, 4)), mode="constant", constant_values=0)

    # Normalize to [0,1] and add dims
    padded = padded.astype(np.float32) / 255.0
    padded = padded.reshape(1, 28, 28, 1)
    return padded


def preprocess_drawn(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _preprocess_core(img)


def preprocess_uploaded(img, invert):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Uploaded image may have black digit on white; allow user to invert
    if invert:
        img = 255 - img
    return _preprocess_core(img)


def predict(img_array):
    probs = model.predict(img_array, verbose=0)
    return probs


def _is_valid_x(x):
    return (
        x is not None and hasattr(x, "ndim") and x.ndim == 4 and x.shape == (1, 28, 28, 1)
    )


# Canvas (centered using columns)
left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    canvas = st_canvas(
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# From canvas
if canvas.image_data is not None:
    rgba = canvas.image_data.astype(np.uint8)
    if rgba.size == 0:
        rgba = None
    if rgba is not None:
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = None
    if gray is not None and np.max(gray) > 0:  # user drew something
        x = preprocess_drawn(gray)
        if not _is_valid_x(x):
            st.warning("No digit detected. Try drawing larger.")
        else:
            if st.checkbox("Show processed image", value=False, key="show_processed_drawn"):
                st.image((x[0, :, :, 0] * 255).astype(np.uint8), caption="Processed (28x28)", width=200, clamp=True)
            probs = predict(x)
            # Expect probs shape (1, 10)
            if probs is None or getattr(probs, "ndim", 0) != 2 or probs.shape[1] != 10:
                st.warning("Unexpected model output. Please try again.")
            else:
                p = probs[0]
                st.write(f"Prediction: {int(np.argmax(p))}")
                st.bar_chart(p)

# Uploader
uploaded = st.file_uploader("Or upload an image (png/jpg)", type=["png", "jpg", "jpeg"])
invert_upload = st.checkbox(
        "Invert uploaded image colors (if background is light)", value=False, key="invert_upload")

# From upload
if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("Could not read the uploaded image. Please try a different file.")
    else:
        st.image(img, caption="Uploaded image", width=300)
        x = preprocess_uploaded(img, invert_upload)
        if not _is_valid_x(x):
            st.warning("No digit detected in the uploaded image. Try inverting or using a clearer image.")
        else:
            if st.checkbox("Show processed image", value=False, key="show_processed_upload"):
                st.image((x[0, :, :, 0] * 255).astype(np.uint8), caption="Processed (28x28)", width=200, clamp=True)
            probs = predict(x)
            if probs is None or getattr(probs, "ndim", 0) != 2 or probs.shape[1] != 10:
                st.warning("Unexpected model output. Please try again.")
            else:
                p = probs[0]
                st.write(f"Prediction: {int(np.argmax(p))}")
                st.bar_chart(p)
