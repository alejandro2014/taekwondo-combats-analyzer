from ultralytics import YOLO

def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
    except Exception as ex:
        print(f"[ERROR] Unable to load model. Check the specified path: {model_path}")
        print(ex)

    return model