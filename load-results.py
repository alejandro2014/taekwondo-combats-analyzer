from datetime import datetime

import argparse
import joblib

from model_loader import load_yolo_model

from video_data_extractor import VideoDataExtractor

def get_arguments():
    parser = argparse.ArgumentParser(description='Extracts information of the provided video')

    parser.add_argument('--input-video', help='Name of the video')
    parser.add_argument('--yolo-model', default='yolov8n-pose.pt', help='YOLO model to use')

    return parser.parse_args()

def get_video_output_name(input_video):
    now = datetime.now()

    timestamp = now.strftime("%Y%m%d-%H%M%S")

    return f"{input_video.split('.')[0]}-{timestamp}.sav"

args = get_arguments()

input_video_path = f'videos/{args.input_video}'
output_video_path = f'videos/output/{get_video_output_name(args.input_video)}'

yolo_model_path = f'weights/{args.yolo_model}'
yolo_model = load_yolo_model(yolo_model_path)

data_extractor = VideoDataExtractor(yolo_model)

video_information = data_extractor.get_video_information(input_video_path)

joblib.dump(video_information, output_video_path)

exit()

def load_video_info(info_path):
    return joblib.load(info_path)

vid_info = load_video_info('combat-capture.sav')

import joblib
import numpy as np

frames = joblib.load('combat-capture.sav')

print(len(frames))
exit()

def get_difference(fighter1, fighter2):
    return np.mean(np.abs(fighter1 - fighter2))

def get_fighters_for_frame(frames, frame_num):
    current_frame = np.array(frames[frame_num])
    previous_frame = np.array(frames[frame_num - 1])

    fighter_1_current = current_frame[0]

    diff1 = get_difference(fighter_1_current, previous_frame[0])
    diff2 = get_difference(fighter_1_current, previous_frame[1])

    indices = [0, 1] if diff1 > diff2 else [1, 0]

    return [ previous_frame[i].tolist() for i in indices ]

start_frame = 11
end_frame = 13

results = [ get_fighters_for_frame(frames, i) for i in range(start_frame, end_frame + 1) ]

print(results)