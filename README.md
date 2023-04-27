# Image And Video Object Detection Using YOLOv8 & Streamlit

This is a web application for object detection using YOLOv8 and Streamlit. Users can upload an image, select a confidence threshold for the model, and download the resulting image with objects detected.

## Tracking With Object Detection Demo



https://user-images.githubusercontent.com/104087274/234874398-75248e8c-6965-4c91-9176-622509f0ad86.mov



## Demo Pics

### Home page

<img src="https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/blob/master/assets/pic1.png" >

### Page after uploading an image and object detection

<img src="https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/blob/master/assets/pic3.png" >

### Segmentation task on image

<img src="https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/blob/master/assets/segmentation.png" >

## Requirements

Python 3.6+
YOLOv8
Streamlit

```bash
pip install ultralytics streamlit
```

## Installation

- Clone the repository: git clone <https://github.com/CodingMantras/yolov8-streamlit-detection-tracking.git>
- Change to the repository directory: `cd yolov8-streamlit-detection-tracking`
- Install the requirements: `pip install -r requirements.txt`
- Download the pre-trained YOLOv8 weights from (<https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt>) and save them to the `weights` directory in the same project.

## Usage

- Run the app with the following command: `streamlit run app.py`
- The app should open in a new browser window.

### ML Model Config

- Select task (Detection, Segmentation)
- Select model confidence
- Use the slider to adjust the confidence threshold (25-100) for the model.

One the model config is done, select a source.

### Detection on images

- The default image with its objects-detected image is displayed on the main page.
- Select a source. (radio button selection `Image`).
- Upload an image by clicking on the "Browse files" button.
- Click the "Detect Objects" button to run the object detection algorithm on the uploaded image with the selected confidence threshold.
- The resulting image with objects detected will be displayed on the page. Click the "Download Image" button to download the image.("If save image to download" is selected)

## Detection in Videos

- Create a folder with name `videos` in the same directory
- Dump your videos in this folder
- In `settings.py` edit the following lines.

```python
# video
VIDEO_DIR = ROOT / 'videos' # After creating the videos folder

# Suppose you have four videos inside videos folder
# Edit the name of video_1, 2, 3, 4 (with the names of your video files) 
VIDEO_1_PATH = VIDEO_DIR / 'video_1.mp4' 
VIDEO_2_PATH = VIDEO_DIR / 'video_2.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'video_3.mp4'
VIDEO_4_PATH = VIDEO_DIR / 'video_4.mp4'

# Edit the same names here also.
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH,
    'video_3': VIDEO_3_PATH,
    'video_4': VIDEO_4_PATH,
}

# Your videos will start appearing inside streamlit webapp 'Choose a video'.
```

- Click on `Detect Video Objects` button and the selected task (detection/segmentation) will start on the selected video.

### Detection on RTSP

- Select the RTSP stream button
- Enter the rtsp url inside the textbox and hit `Detect Objects` button

### Detection on YouTube Video URL

- Select the source as YouTube
- Copy paste the url inside the text box.
- The detection/segmentation task will start on the YouTube video url

<https://user-images.githubusercontent.com/104087274/226178296-684ad72a-fe5f-4589-b668-95c835cd8d8a.mov>

## Acknowledgements

This app is based on the YOLOv8(<https://github.com/ultralytics/ultralytics>) object detection algorithm. The app uses the Streamlit(<https://github.com/streamlit/streamlit>) library for the user interface.

**Hit star ⭐ if you like this repo!!!**
