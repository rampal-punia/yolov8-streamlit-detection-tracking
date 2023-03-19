from pathlib import Path
import PIL

import streamlit as st
import torch
import cv2

import settings
import helper


def stream_camera(source_camera):
    vid_cap = cv2.VideoCapture(source_camera)
    stframe = st.empty()
    while (vid_cap.isOpened()):
        success, image = vid_cap.read()
        if success:
            image = cv2.resize(image, (720, int(720*(9/16))))
            res = model.predict(image, conf=conf)
            res_plotted = res[0].plot()
            stframe.image(res_plotted,
                          caption='Detected Video',
                          channels="BGR",
                          use_column_width=True
                          )


# Sidebar
st.title("Object Detection using YOLOv8")

st.sidebar.header("ML Model Config")
mlmodel_radio = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])
conf = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
if mlmodel_radio == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif mlmodel_radio == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)
model = helper.load_model(model_path)

source_img = None
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", ['Image', 'Video', 'Webcam', 'RTSP'])
if source_radio == 'Image':
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    save_radio = st.sidebar.radio("Save image to download", ["Yes", "No"])
    save = True if save_radio == 'Yes' else False
elif source_radio == 'Video':
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())
elif source_radio == 'Webcam':
    source_path = 0
elif source_radio == 'RTSP':
    source_path = st.sidebar.text_input("rtsp stream url")


detect_button = st.sidebar.button('Detect Objects')

# body
if source_radio == 'Image':
    col1, col2 = st.columns(2)

    with col1:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption='Default Image',
                     use_column_width=True)
        else:
            image = PIL.Image.open(source_img)
            st.image(source_img, caption='Uploaded Image',
                     use_column_width=True)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if detect_button:
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

elif source_radio == 'Video':
    video_file = open(settings.VIDEOS_DICT.get(source_vid), 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    if detect_button:
        vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
        stframe = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                image = cv2.resize(image, (720, int(720*(9/16))))
                res = model.predict(image, conf=conf)
                res_plotted = res[0].plot()
                stframe.image(res_plotted,
                              caption='Detected Video',
                              channels="BGR",
                              use_column_width=True
                              )
elif source_radio == 'Webcam' or source_radio == 'RTSP':
    if detect_button:
        print(source_path)
        print(type(source_path))
        vid_cap = cv2.VideoCapture(source_path)
        stframe = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                image = cv2.resize(image, (720, int(720*(9/16))))
                res = model.predict(image, conf=conf)
                res_plotted = res[0].plot()
                stframe.image(res_plotted,
                              caption='Detected Video',
                              channels="BGR",
                              use_column_width=True
                              )
