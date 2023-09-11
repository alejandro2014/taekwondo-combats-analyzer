import cv2
import json

import streamlit as st

from file_loader import FileLoader

class GUI:
    def __init__(self, combat_analyzer, language='es'):
        self.file_loader = FileLoader()
        self.combat_analyzer = combat_analyzer
        self.language = language
        
    def init(self):
        self.msg = self.load_language()

        self.configure_page()
        self.configure_sidebar()

        st.title(self.msg['app_title'])

        chosen_video = st.session_state.video_chose

        self.show_original_video()

        if st.session_state.detect_objects_button:
            vid_cap = cv2.VideoCapture(chosen_video)

            st_frame = st.empty()

            results = self.get_results(vid_cap)

    def load_language(self):
        with open('languages.json') as json_file:
            file_contents = json_file.read()

            msg_languages = json.loads(file_contents)

        return msg_languages[self.language]
    
    def configure_page(self):
        st.set_page_config(
            page_title = self.msg['app_title'],
            page_icon = "🤼‍♂️",
            layout = "wide",
            initial_sidebar_state = "expanded"
        )

    def configure_sidebar(self):
        with st.sidebar:
            st.header(self.msg['sidebar_header'])
        
            st.selectbox(self.msg['video'], self.file_loader.get_available_videos(), key="video_chose")
            st.selectbox(self.msg['model'], self.file_loader.get_available_models(), key="model_chose")

            st.slider(self.msg['confidence'], 25, 100, 50, key='chosen_confidence')

            st.checkbox(self.msg['show_original'], key='show_original_video')
            st.checkbox(self.msg['show_analyzed'], key='show_analyzed_video')

            st.button(self.msg['detect'], key='detect_objects_button')

    def show_original_video(self):
        show_video = st.session_state.show_original_video

        if not show_video:
            return
        
        source_vid = st.session_state.video_chose

        with open(source_vid, 'rb') as video_file:
            video_bytes = video_file.read()

        st.video(video_bytes)

    def get_results(self, vid_cap):
        results = []

        while (vid_cap.isOpened()):
            success, image = vid_cap.read()

            if success:
                res, res_plotted = self.generate_image(image)
                results.append(res)

                show = st.session_state.show_analyzed_video

                if show:
                    st_frame.image(res_plotted, channels="BGR", use_column_width=True)
            else:
                vid_cap.release()
                break

        return results
    
    def generate_image(self, image):
        image = cv2.resize(image, (720, int(720*(9/16))))
        confidence = float(st.session_state.chosen_confidence) / 100

        res = self.combat_analyzer.predict(image, confidence)
        res_plotted = res[0].plot()

        return res, res_plotted
    
    def load_model(self):
        model_path = st.session_state.model_chose

        self.combat_analyzer.load_model(model_path)

        try:
            model = YOLO(model_path)
        except Exception as ex:
            st.error(f"Unable to load model. Check the specified path: {model_path}")
            st.error(ex)

        return model