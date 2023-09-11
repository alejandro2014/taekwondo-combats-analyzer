import cv2

class VideoDataExtractor:
    def __init__(self, model, confidence=0.4, visual=False, st_frame=None):
        self.model = model
        self.confidence = confidence
        self.visual = visual
        self.st_frame = st_frame

    def get_video_information(self, video_path):
        video_capture = self.get_video_capture(video_path)

        return {
            'path': video_path,
            'properties': self.get_properties(video_capture),
            'results': self.get_results(video_capture, self.model)
        }
    
    def get_properties(self, video_capture):
        properties_names = [ 'FPS', 'FRAME_COUNT', 'FRAME_HEIGHT', 'FRAME_WIDTH' ]

        return {
            property: video_capture.get(getattr(cv2, f'CAP_PROP_{property}'))
            for property in properties_names
        }
    
    def get_results(self, video_capture, yolo_model):
        results = []

        while (video_capture.isOpened()):
            success, image = video_capture.read()

            if not success:
                video_capture.release()
                break

            res = self.generate_frame_result(image, yolo_model)
            results.append(res)

        return results
    
    def generate_frame_result(self, image, model):
        image = self.resize_frame(image)
        
        result = model.predict(image, conf=self.confidence)

        result_info = result[0].keypoints.xyn.tolist()

        if self.visual:
            self.st_frame.image(result[0].plot(), channels="BGR", use_column_width=True)
            
        return  result_info

    def get_video_capture(self, video_path):
        return cv2.VideoCapture(video_path)

    def resize_frame(self, image, new_width=720, video_ratio=(9/16)):
        new_height = int(new_width * video_ratio)
        new_dimensions = (new_width, new_height)

        return cv2.resize(image, new_dimensions)