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

- Run the app with the following command: `streamlit run app.py`
- The app should open in a new browser window.
- The default image with its objects-detected image is displayed on the main page.
- Select a source. **Currently only image source(more features to follow)**. Upload an image by clicking on the "Browse files" button.
- Use the slider to adjust the confidence threshold for the model.
- Click the "Detect Objects" button to run the object detection algorithm on the uploaded image with the selected confidence threshold.
- The resulting image with objects detected will be displayed on the page. Click the "Download Image" button to download the image.("If save image to download" is selected)

## Acknowledgements

This app is based on the YOLOv8(<https://github.com/ultralytics/ultralytics>) object detection algorithm. The app uses the Streamlit(<https://github.com/streamlit/streamlit>) library for the user interface.
