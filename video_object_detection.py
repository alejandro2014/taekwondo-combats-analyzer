import cv2
import os
import streamlit as st

from ultralytics import YOLO

MODELS_DIR = 'weights'
VIDEOS_DIR = 'videos'

title = "Taekwondo combats analyzer"

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
    
        st.sidebar.selectbox("Video", get_available_videos(), key="video_chose")
        st.sidebar.selectbox("Model", get_available_models(), key="model_chose")

        st.slider("Select Model Confidence", 25, 100, 40, key='chosen_confidence')

def get_available_videos():
    return [ f'{VIDEOS_DIR}/{video}' for video in os.listdir(f'{VIDEOS_DIR}/') ]

def get_available_models():
    return [ f'{MODELS_DIR}/{model}' for model in os.listdir(f'{MODELS_DIR}/') ]

def load_model(model_path):
    try:
        model = YOLO(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    return model

def generate_image(image):
    image = cv2.resize(image, (720, int(720*(9/16))))
    confidence = float(st.session_state.chosen_confidence) / 100

    res = model.predict(image, conf=confidence)
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

configure_sidebar()

st.title(title)

chosen_model = st.session_state.model_chose
chosen_video = st.session_state.video_chose

model = load_model(chosen_model)

video_bytes = get_video_bytes(chosen_video)

if video_bytes:
    st.video(video_bytes)

if st.sidebar.button('Detect Objects'):
    vid_cap = cv2.VideoCapture(chosen_video)
    
    st_frame = st.empty()
    
    results = show_video(vid_cap, show=True)