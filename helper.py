from ultralytics import YOLO


def load_model(model_path):
    model = YOLO(model_path)
    return model
