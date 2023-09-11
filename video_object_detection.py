import joblib
import json
import os
import streamlit as st

from model_loader import load_yolo_model

from video_data_extractor import VideoDataExtractor

LANGUAGE = 'es'
MODELS_DIR = 'weights'
VIDEOS_DIR = 'videos'

def load_messages(language):
    with open('languages.json') as json_file:
        file_contents = json_file.read()

    msg_languages = json.loads(file_contents)

    return msg_languages[language]

def get_available_files(root_dir):
    return [ f'{root_dir}/{file}' for file in os.listdir(f'{root_dir}/') ]

def get_available_videos():
    return get_available_files(VIDEOS_DIR)

def get_available_models():
    return get_available_files(MODELS_DIR)
    
def configure_page():
    st.set_page_config(
        page_title = MSG['app_title'],
        page_icon = "🤼‍♂️",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )

    st.title(MSG['app_title'])

def configure_sidebar():
    with st.sidebar:
        st.header(MSG['sidebar_header'])
    
        st.selectbox(MSG['video'], get_available_videos(), key="video_chose")
        st.selectbox(MSG['model'], get_available_models(), key="model_chose")

        st.slider(MSG['confidence'], 25, 100, 40, key='chosen_confidence')

        st.checkbox(MSG['show_original'], key='show_original_video')
        st.checkbox(MSG['show_analyzed'], key='show_analyzed_video')

        st.button(MSG['detect'], key='detect_objects_button')

def show_original_video():
    show_video = st.session_state.show_original_video

    if not show_video:
        return
    
    source_vid = st.session_state.video_chose

    with open(source_vid, 'rb') as video_file:
        video_bytes = video_file.read()

    st.video(video_bytes)

MSG = load_messages(LANGUAGE)

configure_page()
configure_sidebar()

video_path = st.session_state.video_chose
yolo_model = load_yolo_model(st.session_state.model_chose)

show_original_video()

if st.session_state.detect_objects_button:
    st_frame = st.empty()

    data_extractor = VideoDataExtractor(yolo_model, visual=True, st_frame=st_frame)

    video_information = data_extractor.get_video_information(video_path)

    joblib.dump(video_information, 'combat-capture.sav')