import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("📸 HSV Object Detection on Captured Image")

# Layout: Create columns for the images and sliders
col1, col2 = st.columns(2)

# HSV sliders in the sidebar for better layout
l_h = st.sidebar.slider("Lower H", 0, 179, 0)
l_s = st.sidebar.slider("Lower S", 0, 255, 0)
l_v = st.sidebar.slider("Lower V", 0, 255, 0)

u_h = st.sidebar.slider("Upper H", 0, 179, 179)
u_s = st.sidebar.slider("Upper S", 0, 255, 255)
u_v = st.sidebar.slider("Upper V", 0, 255, 255)

# Capture image input
camera_image = st.camera_input("📷 Capture Image")

if camera_image:
    # Read the image from the camera
    img = Image.open(camera_image)
    frame = np.array(img)

    # Convert to HSV
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

