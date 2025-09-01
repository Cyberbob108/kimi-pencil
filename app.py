# sketch_app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ------------------------------------------------------------------
# 1. Streamlit page configuration
# ------------------------------------------------------------------
st.set_page_config(page_title="Realistic Pencil-Sketch Generator",
                   layout="centered")

st.title("🖼️ → ✏️ Realistic Pencil-Sketch Generator")
st.write("Upload an image and watch it become a hand-drawn-style sketch.")

# ------------------------------------------------------------------
# 2. Helper: Create the sketch
# ------------------------------------------------------------------
def make_pencil_sketch(image_rgb: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    image_rgb : np.ndarray
        Input image in RGB channel order (height, width, 3).

    Returns
    -------
    np.ndarray
        Sketch image in RGB channel order, same size as input.
    """
    # 2.1 Convert to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 2.2 Invert the grayscale image
    inverted = 255 - gray

    # 2.3 Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, ksize=(21, 21), sigmaX=0, sigmaY=0)

    # 2.4 Color-dodge blending (element-wise)
    # The dodge formula:  sketch = min(255, (base * 255) / (255 - blend))
    # Here the "base" is the original grayscale, "blend" is the blurred+inverted.
    sketch = cv2.divide(gray, 255 - blurred, scale=256)

    # 2.5 Convert single-channel sketch back to 3-channel for display
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    return sketch_rgb

# ------------------------------------------------------------------
# 3. Streamlit file uploader
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose an image (jpg, jpeg, png)",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image bytes -> PIL -> RGB numpy array
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(pil_img)

    # ------------------------------------------------------------------
    # 4. Generate sketch
    # ------------------------------------------------------------------
    with st.spinner("Sketching..."):
        sketch_rgb = make_pencil_sketch(img_rgb)

    # ------------------------------------------------------------------
    # 5. Display side by side
    # ------------------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(img_rgb, use_column_width=True)
    with col2:
        st.subheader("Pencil Sketch")
        st.image(sketch_rgb, use_column_width=True)

    # ------------------------------------------------------------------
    # 6. Optional: Download button
    # ------------------------------------------------------------------
    sketch_pil = Image.fromarray(sketch_rgb)
    buf = io.BytesIO()
    sketch_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(label="Download Sketch",
                       data=byte_im,
                       file_name="pencil_sketch.png",
                       mime="image/png")
