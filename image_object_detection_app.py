import PIL

import streamlit as st
from ultralytics import YOLO

title = "Taekwondo combats analyzer"
model_path = 'weights/yolov8n-pose.pt'
image_extensions = ("jpg", "jpeg", "png", 'bmp', 'webp')

st.set_page_config(
    page_title=title,
    page_icon="🤼‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("Image/Video Config")
    
    source_img = st.file_uploader("Choose an image...", type=image_extensions)

    confidence = float(st.slider("Model Confidence", 25, 100, 40)) / 100

st.title(title)

col1, col2 = st.columns(2)

with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        
        st.image(source_img, caption="Uploaded Image", use_column_width=True)

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image, conf=confidence)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]

    with col2:
        st.image(res_plotted, caption='Detected Image', use_column_width=True)
        
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet!")