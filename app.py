from pathlib import Path
import PIL

import streamlit as st
import torch
from ultralytics import YOLO

import settings


def load_model(model_path):
    model = YOLO(model_path)
    return model


st.title("Object Detection using YOLOv8")


with st.sidebar:
    st.header("Image/Video Config")
    source_radio = st.radio(
        "Select Source", ['Image', 'Video', 'Webcam', 'RTSP'])
    if source_radio == 'Image':
        source = st.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    elif source_radio == 'Video':
        source = st.file_uploader(
            "Choose a video...", type=("mp4", "mpeg", "avi", 'gif', 'mov', 'm4v'))
    elif source_radio == 'Webcam':
        source = st.file_uploader(
            "Choose a video...", type=("mp4", "mpeg", "avi", 'gif', 'mov', 'm4v'))
    elif source_radio == 'RTSP':
        source = st.text_input("rtsp stream url", )

    save_radio = st.radio("Save image to download", ["Yes", "No"])

    st.header("ML Model Config")
    mlmodel_radio = st.radio(
        "Select Task", ['Detection', 'Segmentation'])
    conf = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100
    print(conf)
    detect_button = st.button('Detect Objects')

col1, col2 = st.columns(2)

with col1:
    if source is None:
        default_image_path = str(settings.DEFAULT_IMAGE)
        image = PIL.Image.open(default_image_path)
        st.image(default_image_path, caption='Default Image',
                 use_column_width=True)
    else:
        image = PIL.Image.open(source)
        st.image(source, caption='Uploaded Image',
                 use_column_width=True)


with col2:
    if source is None:
        default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
        image = PIL.Image.open(default_detected_image_path)
        st.image(default_detected_image_path, caption='Detected Image',
                 use_column_width=True)
    else:
        if detect_button:
            if mlmodel_radio == 'Detection':
                model_path = Path(settings.DETECTION_MODEL)
            elif mlmodel_radio == 'Segmentation':
                model_path = Path(settings.SEGMENTATION_MODEL)
            save = True if save_radio == 'Yes' else False

            model = load_model(model_path)
            with torch.no_grad():
                res = model.predict(
                    image, save=save, save_txt=save, exist_ok=True, conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                with open('runs/detect/predict/image0.jpg', 'rb') as fl:
                    st.download_button("Download object-detected image",
                                       data=fl,
                                       file_name="image0.jpg",
                                       mime='image/jpg'
                                       )
if detect_button:
    with st.expander("Detection Results"):
        for box in boxes:
            st.write(box.xywh)
