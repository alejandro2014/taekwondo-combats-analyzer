import cv2
import streamlit as st
from ultralytics import YOLO

title = "Taekwondo combats analyzer"

#model_path = "weights/yolov8n.pt"
model_path = 'weights/yolov8n-pose.pt'

def configure_page(title):
    st.set_page_config(
        page_title=title,
        page_icon="🤼‍♂️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def configure_sidebar():
    with st.sidebar:
        st.header("Image/Video Config")
    
        source_vid = st.sidebar.selectbox("Choose a video...", get_available_videos())

        confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100

    return source_vid, confidence

def get_available_videos():
    return [
        "videos/combat.mp4",
        "videos/office-video.mp4"
    ]

def load_model(model_path):
    try:
        model = YOLO(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    return model

def generate_image(image):
    image = cv2.resize(image, (720, int(720*(9/16))))

    res = model.predict(image, conf=confidence)

    #result_tensor = res[0].boxes
    res_plotted = res[0].plot()

    return res, res_plotted

def get_video_bytes(source_vid):
    with open(str(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()

    return video_bytes

def show_video(vid_cap, show=True):
    results = []

    while (vid_cap.isOpened()):
        success, image = vid_cap.read()

        if success:
            res, res_plotted = generate_image(image)
            results.append(res)

            if show:
                st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)
        else:
            vid_cap.release()
            break

    return results

configure_page(title)

source_vid, confidence = configure_sidebar()

st.title(title)

model = load_model(model_path)

if source_vid is None:
    exit()

video_bytes = get_video_bytes(source_vid)

if video_bytes:
    st.video(video_bytes)

if st.sidebar.button('Detect Objects'):
    vid_cap = cv2.VideoCapture("videos/combat.mp4")
    st_frame = st.empty()
    
    results = show_video(vid_cap, show=False)

    print(results)