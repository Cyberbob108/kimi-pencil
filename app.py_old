import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Pencil Sketch", layout="centered")
st.title("🖼️ → ✏️ Realistic Pencil-Sketch Generator")

@st.cache_data
def pencil_sketch(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img_rgb = np.array(Image.open(uploaded).convert("RGB"))
    sketch_rgb = pencil_sketch(img_rgb)

    col1, col2 = st.columns(2)
    col1.image(img_rgb, caption="Original", use_column_width=True)
    col2.image(sketch_rgb, caption="Sketch", use_column_width=True)

    buf = io.BytesIO()
    Image.fromarray(sketch_rgb).save(buf, format="PNG")
    st.download_button("⬇️ Download Sketch",
                       data=buf.getvalue(),
                       file_name="sketch.png",
                       mime="image/png")
