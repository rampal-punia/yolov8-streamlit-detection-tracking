# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
import base64

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.markdown("<h1 style='text-align: center;'> Welcome to the object detection and tracking program with YOLOv8 :) </h1>", unsafe_allow_html=True)


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Sidebar
st.sidebar.header("ML Model Config")


confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 50)) / 100


model=None
# loading weight file
weight_file = st.sidebar.file_uploader("Upload Model Weight File", type=("pt"))

# loading weight file and creat file
if weight_file:
    model_path = Path(weight_file.name)
    try:
        model = helper.load_model(model_path)
        st.success("Model successfully loaded.")
    except Exception as ex:
        st.error(f"Unable to load model. Error: {ex}")


st.sidebar.header("Image/Video Config")
source_radio = st.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                set_background('back_ground_images/background1.jpg')
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            pass
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    set_background('back_ground_images/background2.jpg')
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    set_background('back_ground_images/background3.jpg')
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    set_background('back_ground_images/background4.jpg')
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    set_background('back_ground_images/background5.jpg')
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
