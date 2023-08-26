import streamlit as st
import PIL

st.set_page_config(
    page_title="Taekwondo combats analyzer",
    page_icon="🤼‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

image_extensions = ("jpg", "jpeg", "png", 'bmp', 'webp')

with st.sidebar:
    st.header("Image/Video Config")
    
    source_img = st.sidebar.file_uploader("Choose an image...", type=image_extensions)

st.title("Taekwondo combats analyzer")

col1, col2 = st.columns(2)

with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img)

        st.image(source_img, caption="Uploaded Image", use_column_width=True)