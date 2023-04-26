from ultralytics import YOLO
import streamlit as st
import cv2
import pafy
import numpy as np

from sort import Sort
import settings

# SORT tracking algorithm initialization
sort_max_age = 5
sort_min_hits = 2
sort_iou_thresh = 0.2
sort_tracker = Sort(max_age=sort_max_age,
                    min_hits=sort_min_hits,
                    iou_threshold=sort_iou_thresh)
track_color_id = 0

# Sorting algorithm
'''Computer Color for every box and track'''
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    color = [int(int(p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


"""" Calculates the relative bounding box from absolute pixel values. """


def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


"""Function to Draw Bounding boxes"""


def draw_boxes(img, bbox, identities=None, categories=None,
               names=None, color_box=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2), (int((box[1]+box[3])/2)))
        label = str(id)

        if color_box:
            color = compute_color_for_labels(id)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 191, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        [255, 255, 255], 1)
            cv2.circle(img, data, 3, color, -1)
        else:
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 191, 0), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 191, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        [255, 255, 255], 1)
            cv2.circle(img, data, 3, (255, 191, 0), -1)
    return img


def load_model(model_path):
    model = YOLO(model_path)
    return model


def _display_detected_trackes(dets, img, color_box=None):
    # pass an empty array to sort
    dets_to_sort = np.empty((0, 6))

    # NOTE: We send in detected object class too
    for x1, y1, x2, y2, conf, detclass in dets.cpu().detach().numpy():
        dets_to_sort = np.vstack((dets_to_sort,
                                  np.array([x1, y1, x2, y2,
                                            conf, detclass])))

    # Run SORT
    tracked_dets = sort_tracker.update(dets_to_sort)
    tracks = sort_tracker.getTrackers()

    # loop over tracks
    for track in tracks:
        if color_box:
            color = compute_color_for_labels(track_color_id)
            [cv2.line(img, (int(track.centroidarr[i][0]), int(track.centroidarr[i][1])),
                      (int(track.centroidarr[i+1][0]),
                       int(track.centroidarr[i+1][1])),
                      color, thickness=3) for i, _ in enumerate(track.centroidarr)
                if i < len(track.centroidarr)-1]
            track_color_id = track_color_id+1
        else:
            [cv2.line(img, (int(track.centroidarr[i][0]), int(track.centroidarr[i][1])),
                      (int(track.centroidarr[i+1][0]),
                       int(track.centroidarr[i+1][1])),
                      (124, 252, 0), thickness=3) for i, _ in enumerate(track.centroidarr)
                if i < len(track.centroidarr)-1]

    # draw boxes for visualization
    if len(tracked_dets) > 0:
        bbox_xyxy = tracked_dets[:, :4]
        identities = tracked_dets[:, 8]
        categories = tracked_dets[:, 4]
        names = None
        draw_boxes(img, bbox_xyxy, identities,
                   categories, names, color_box)


def _display_detected_frames(conf, model, stframe, image, is_display_tracking=None):
    image = cv2.resize(image, (720, int(720*(9/16))))
    res = model.predict(image, conf=conf)
    if is_display_tracking:
        _display_detected_trackes(dets)
    res_plotted = res[0].plot()
    # st.write(dir(res))
    stframe.image(res_plotted,
                  caption='Detected Video',
                  channels="BGR",
                  use_column_width=True
                  )


def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video url")
    if st.sidebar.button('Detect Objects'):
        try:
            video = pafy.new(source_youtube)
            best = video.getbest(preftype="mp4")
            cap = cv2.VideoCapture(best.url)
            stframe = st.empty()
            while (cap.isOpened()):
                success, image = cap.read()
                if success:
                    _display_detected_frames(conf, model, stframe, image)
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    source_rtsp = st.sidebar.text_input("rtsp stream url")
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            stframe = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, stframe, image)
        except Exception as e:
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    source_webcam = settings.WEBCAM_PATH
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            stframe = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, stframe, image)
        except Exception as e:
            st.sidebar.error("Error loading webcam: " + str(e))


def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False

    video_file = open(settings.VIDEOS_DICT.get(source_vid), 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        # try:
        vid_cap = cv2.VideoCapture(
            str(settings.VIDEOS_DICT.get(source_vid)))
        stframe = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf, model, stframe, image, is_display_tracker)
        # except Exception as e:
        #     st.sidebar.error("Error loading video: " + str(e))
