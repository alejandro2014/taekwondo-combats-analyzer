import cv2
import joblib
import json
import os
import streamlit as st

from ultralytics import YOLO

MODELS_DIR = 'weights'
VIDEOS_DIR = 'videos'

LANGUAGE = 'es'

with open('languages.json') as json_file:
    file_contents = json_file.read()

MSG_LANGUAGES = json.loads(file_contents)

MSG = MSG_LANGUAGES[LANGUAGE]

def configure_page():
    st.set_page_config(
        page_title = MSG['app_title'],
        page_icon = "🤼‍♂️",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )

def configure_sidebar():
    with st.sidebar:
        st.header(MSG['sidebar_header'])
    
        st.selectbox(MSG['video'], get_available_videos(), key="video_chose")
        st.selectbox(MSG['model'], get_available_models(), key="model_chose")

        st.slider(MSG['confidence'], 25, 100, 40, key='chosen_confidence')

        st.checkbox(MSG['show_original'], key='show_original_video')
        st.checkbox(MSG['show_analyzed'], key='show_analyzed_video')

        st.button(MSG['detect'], key='detect_objects_button')

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
    res = res[0].keypoints.xyn.tolist()

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

    show = st.session_state.show_analyzed_video

    while (vid_cap.isOpened()):
        success, image = vid_cap.read()

        if not success:
            vid_cap.release()
            break

        res, res_plotted = generate_image(image)
        results.append(res)

        if show:
            st_frame.image(res_plotted, channels="BGR", use_column_width=True)

    return results

configure_page()

configure_sidebar()

st.title(MSG['app_title'])

chosen_video = st.session_state.video_chose

model = load_model()

show_original_video()

if st.session_state.detect_objects_button:
    vid_cap = cv2.VideoCapture(chosen_video)
    
    st_frame = st.empty()
    
    results = get_results(vid_cap)

    joblib.dump(results, 'combat-capture.sav')