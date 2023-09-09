You can find the necessary information about the source code at this link: [https://github.com/CodingMantras/yolov8-streamlit-detection-tracking.git].

Preliminary Description:

The purpose of this work is to make the code more interactive and functional. Unlike the original code, it can now accept weight files and video sources from external locations without limitations. This allows for working with different weight files or videos without the need to close and reopen the program.

Additionally, an enhancement has been made for Intersection over Union (IoU) within the object tracking section.

Furthermore:
Don't forget to install the following libraries:

## Requirements

Python 3.6+
YOLOv8
Streamlit

```bash
pip install ultralytics streamlit pafy
pip install pytube
```

# Running the program :
- clone repo
```bash
git clone https://github.com/ilyasdemir-demirilyas/yolov8-streamlit-detection-tracking.git
```
- Entering the project directory.
```bash
cd yolov8-streamlit-detection-tracking
```
- Run the dashboard.
``` 
streamlit run app.py
```
- We are now inside the image detection page.
- Now, we can perform detection by uploading the necessary files.

### image page

<img src="https://user-images.githubusercontent.com/80126067/265277736-240e9460-ea6a-4144-8fa4-adedf9e55db0.png" >

### video page

<img src="https://user-images.githubusercontent.com/80126067/265277734-5e833394-dd3d-4d4e-9e24-d24873d67076.png" >
