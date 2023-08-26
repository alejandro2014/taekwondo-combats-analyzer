import cv2
import streamlit as st
from ultralytics import YOLO

title = "Taekwondo combats analyzer"

model_path = "weights/yolov8n.pt"

st.set_page_config(
    page_title=title,
    page_icon="🤼‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("Image/Video Config")
    
    source_vid = st.sidebar.selectbox("Choose a video...", ["videos/office-video.mp4"])

    confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100

st.title(title)

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if source_vid is not None:
    with open(str(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()

    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Objects'):
        vid_cap = cv2.VideoCapture("videos/office-video.mp4")
        st_frame = st.empty()
        
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()

            if success:
                image = cv2.resize(image, (720, int(720*(9/16))))
                res = model.predict(image, conf=confidence)
                result_tensor = res[0].boxes
                res_plotted = res[0].plot()
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )
            else:
                vid_cap.release()
                break