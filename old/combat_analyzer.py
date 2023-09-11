from ultralytics import YOLO

class CombatAnalyzer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image, confidence):
        return self.model.predict(image, conf=confidence)