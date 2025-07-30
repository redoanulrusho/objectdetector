import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

st.title("📸 HSV Object Detection on Captured or Uploaded Image by RUSHO")

# Create columns for images and the filtered result
col1, col2 = st.columns(2)

# HSV sliders in the sidebar for better layout
l_h = st.sidebar.slider("Lower H", 0, 179, 0)
l_s = st.sidebar.slider("Lower S", 0, 255, 0)
l_v = st.sidebar.slider("Lower V", 0, 255, 0)

u_h = st.sidebar.slider("Upper H", 0, 179, 179)
u_s = st.sidebar.slider("Upper S", 0, 255, 255)
u_v = st.sidebar.slider("Upper V", 0, 255, 255)

# Image input: Option for capturing or uploading
image_option = st.radio("Choose Image Source", ("Capture Image", "Upload Image"))

# Initialize the image variable
img = None
frame = None

if image_option == "Capture Image":
    camera_image = st.camera_input("📷 Capture Image")
    if camera_image:
        img = Image.open(camera_image)
        frame = np.array(img)
elif image_option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        frame = np.array(img)

# If an image is loaded, process it for object detection
if img is not None and frame is not None:
    # Convert the uploaded/captured image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Define HSV range
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # Create mask based on HSV range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply the mask on the original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Layout: Display images side by side
    with col1:
        st.image(frame, caption="Original Image", channels="RGB")
        st.image(mask, caption="Mask", channels="GRAY")

    with col2:
        st.image(res, caption="Filtered Result", channels="RGB")

    # Add button to save filtered result
    save_button = st.button("Save Filtered Result")

    if save_button:
        # Save the filtered result as a PNG file
        filtered_result = Image.fromarray(res)
        # Create a BytesIO object to save the image in memory
        img_bytes = io.BytesIO()
        filtered_result.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Provide the download link
        st.download_button(
            label="Download Filtered Image",
            data=img_bytes,
            file_name="filtered_result.png",
            mime="image/png"
        )
