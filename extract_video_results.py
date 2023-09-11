import argparse
import joblib

from common import load_yolo_model, get_video_output_name
from video_data_extractor import VideoDataExtractor

def get_arguments():
    parser = argparse.ArgumentParser(description='Extracts information of the provided video')

    parser.add_argument('--input-video', help='Name of the video')
    parser.add_argument('--yolo-model', default='yolov8n-pose.pt', help='YOLO model to use')

    return parser.parse_args()

args = get_arguments()

input_video_path = f'videos/{args.input_video}'
output_video_path = f'videos/output/{get_video_output_name(args.input_video)}'

yolo_model_path = f'weights/{args.yolo_model}'
yolo_model = load_yolo_model(yolo_model_path)

data_extractor = VideoDataExtractor(yolo_model)

video_information = data_extractor.get_video_information(input_video_path)

joblib.dump(video_information, output_video_path)