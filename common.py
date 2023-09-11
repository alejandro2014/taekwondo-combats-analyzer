from datetime import datetime
from ultralytics import YOLO

def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
    except Exception as ex:
        print(f"[ERROR] Unable to load model. Check the specified path: {model_path}")
        print(ex)

    return model

def get_video_output_name(input_video):
    now = datetime.now()

    timestamp = now.strftime("%Y%m%d-%H%M%S")

    return f"{input_video.split('.')[0]}-{timestamp}.sav"