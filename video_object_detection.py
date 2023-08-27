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

        st.checkbox('Show original video', key='show_original_video')
        st.checkbox('Show analyzed video', key='show_analyzed_video')

        #if agree:
        #    st.write('Great!')

def get_available_videos():
    return [ f'{VIDEOS_DIR}/{video}' for video in os.listdir(f'{VIDEOS_DIR}/') ]

def get_available_models():
    return [ f'{MODELS_DIR}/{model}' for model in os.listdir(f'{MODELS_DIR}/') ]

def load_model():
    model_path = st.session_state.model_chose

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

def show_original_video():
    show_video = st.session_state.show_original_video

    if not show_video:
        return
    
    source_vid = st.session_state.video_chose

    with open(source_vid, 'rb') as video_file:
        video_bytes = video_file.read()

    st.video(video_bytes)

def get_results(vid_cap):
    results = []

    while (vid_cap.isOpened()):
        success, image = vid_cap.read()

        if success:
            res, res_plotted = generate_image(image)
            results.append(res)

            show = st.session_state.show_analyzed_video

            if show:
                st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)
        else:
            vid_cap.release()
            break

    return results

configure_page(title)

configure_sidebar()

st.title(title)

chosen_video = st.session_state.video_chose

model = load_model()

show_original_video()

if st.sidebar.button('Detect Objects'):
    vid_cap = cv2.VideoCapture(chosen_video)
    
    st_frame = st.empty()
    
    results = get_results(vid_cap)