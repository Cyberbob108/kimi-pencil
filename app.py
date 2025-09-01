# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Pencil Sketch", layout="centered")
st.title("🖼️ → ✏️ Realistic Pencil-Sketch Generator")

@st.cache_data
def clean_sketch(bgr: np.ndarray, grain_strength: float = 0.05) -> np.ndarray:
    """
    Classic dodge-and-burn sketch + optional subtle paper grain.
    Parameters
    ----------
    bgr            : original BGR image
    grain_strength : 0–0.15, amount of paper texture (0 = none)
    """
    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)       # edge-preserving
    inv    = 255 - smooth
    blur   = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(smooth, 255 - blur, scale=256)

    # add ultra-subtle paper texture
    if grain_strength > 0:
        h, w   = sketch.shape
        paper  = np.random.normal(loc=245, scale=4, size=(h, w)).astype(np.uint8)
        paper  = cv2.blur(paper, (3, 3))                  # soften grain
        sketch = cv2.addWeighted(sketch, 1 - grain_strength,
                                 paper, grain_strength, 0)

    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

# ------------------------------------------------------------------
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img_rgb = np.array(Image.open(uploaded).convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    grain = st.sidebar.slider("Subtle paper grain", 0.0, 0.15, 0.05, 0.01)
    sketch_rgb = clean_sketch(img_bgr, grain)

    col1, col2 = st.columns(2)
    col1.image(img_rgb,   caption="Original", use_column_width=True)
    col2.image(sketch_rgb, caption="Clean Sketch", use_column_width=True)

    # Download
    buf = io.BytesIO()
    Image.fromarray(sketch_rgb).save(buf, format="PNG")
    st.download_button("⬇️ Download Sketch",
                       data=buf.getvalue(),
                       file_name="sketch.png",
                       mime="image/png")
