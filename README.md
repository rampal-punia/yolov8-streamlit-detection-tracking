# YOLOv8 Object Detection App using Streamlit

This is a web application for object detection using YOLOv8 and Streamlit. Users can upload an image, select a confidence threshold for the model, and download the resulting image with objects detected.

## Demo Pics

### Home page

<img src="https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/blob/master/assets/pic1.png" >

### Page after uploading an image

<img src="https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/blob/master/assets/pic3.png" >

## Requirements

Python 3.6+
YOLOv8
Streamlit

```bash
pip install ultralytics streamlit
```

## Installation

Clone the repository: git clone <https://github.com/CodingMantras/yolov8-streamlit-detection-tracking.git>
Change to the repository directory: cd repo
Install the requirements: pip install -r requirements.txt
Download the pre-trained YOLOv8 weights from (<https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt>) and save them to the weights directory.

## Usage

Run the app with the following command: streamlit run app.py
The app should open in a new browser window. Upload an image by clicking on the "Upload Image" button.
If no image is uploaded, the default image with its objects-detected image will be displayed on the main page in two columns.
Use the slider to adjust the confidence threshold for the model.
Click the "Detect Objects" button to run the object detection algorithm on the uploaded image with the selected confidence threshold.
The resulting image with objects detected will be displayed on the page. Click the "Download Image" button to download the image.

## Acknowledgements

This app is based on the YOLOv8 object detection algorithm, developed by (<https://github.com/username/repo>). The app uses the Streamlit library for the user interface.
