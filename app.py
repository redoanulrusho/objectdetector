import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.title("📲 Live HSV Object Detection (Front/Back Camera)")

# Camera selection
camera_choice = st.selectbox("📷 Select Camera", ["Front Camera", "Back Camera"])
facing_mode = "user" if camera_choice == "Front Camera" else "environment"

# HSV sliders
l_h = st.slider("Lower H", 0, 179, 0)
l_s = st.slider("Lower S", 0, 255, 0)
l_v = st.slider("Lower V", 0, 255, 0)

u_h = st.slider("Upper H", 0, 179, 179)
u_s = st.slider("Upper S", 0, 255, 255)
u_v = st.slider("Upper V", 0, 255, 255)

# WebRTC Configuration (STUN + TURN)
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},  # Google STUN
            {
                "urls": [
                    "turn:openrelay.metered.ca:80",
                    "turn:openrelay.metered.ca:443",
                    "turn:openrelay.metered.ca:443?transport=tcp"
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    }
)

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Debugging: Check if frame is empty
        if img is None:
            st.error("Video feed is not available.")
        else:
            st.write(f"Video feed shape: {img.shape}")  # Display the dimensions of the frame

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Mask
        lower_bound = np.array([l_h, l_s, l_v])
        upper_bound = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        res = cv2.bitwise_and(img, img, mask=mask)

        return res

# WebRTC Stream with better video constraints
webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"facingMode": facing_mode, "width": 640, "height": 480}, "audio": False},
)
