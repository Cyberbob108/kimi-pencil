# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Pencil Sketch", layout="centered")
st.title("🖼️ → ✏️ Realistic Pencil-Sketch Generator")

@st.cache_data
def realistic_sketch(bgr: np.ndarray,
                     shade_scale: float = 1.4,
                     paper_alpha: float = 0.25) -> np.ndarray:
    """
    Create a more realistic pencil/charcoal sketch.
    Parameters
    ----------
    bgr         : original BGR image (from PIL → numpy)
    shade_scale : >1 deepens blacks (graphite look)
    paper_alpha : 0–1 strength of paper grain overlay
    """
    h, w = bgr.shape[:2]

    # 1. grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 2. edge-preserving smoothing → cleaner shading
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    # 3. classic dodge
    inverted = 255 - smooth
    blurred  = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch   = cv2.divide(smooth, 255 - blurred, scale=256)

    # 4. deepen blacks
    sketch = cv2.multiply(sketch, shade_scale)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)

    # 5. procedural paper texture
    paper = np.random.normal(loc=245, scale=6, size=(h, w)).astype(np.uint8)
    paper = cv2.bilateralFilter(paper, 5, 25, 25)  # soften
    sketch = cv2.addWeighted(sketch, 1 - paper_alpha, paper, paper_alpha, 0)

    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

# ------------------------------------------------------------------
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img_rgb = np.array(Image.open(uploaded).convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # UI knobs
    shade = st.sidebar.slider("Shade intensity", 1.0, 2.0, 1.4, 0.05)
    grain = st.sidebar.slider("Paper texture", 0.0, 0.5, 0.25, 0.05)

    sketch_rgb = realistic_sketch(img_bgr, shade, grain)

    col1, col2 = st.columns(2)
    col1.image(img_rgb,  caption="Original", use_column_width=True)
    col2.image(sketch_rgb, caption="Realistic Sketch", use_column_width=True)

    # Download button
    buf = io.BytesIO()
    Image.fromarray(sketch_rgb).save(buf, format="PNG")
    st.download_button("⬇️ Download Sketch",
                       data=buf.getvalue(),
                       file_name="sketch.png",
                       mime="image/png")
