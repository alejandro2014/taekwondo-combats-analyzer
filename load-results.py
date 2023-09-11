import cv2
import joblib

from ultralytics import YOLO

from video_data_extractor import VideoDataExtractor

def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
    except Exception as ex:
        print(f"[ERROR] Unable to load model. Check the specified path: {model_path}")
        print(ex)

    return model

yolo_model = load_yolo_model('weights/yolov8n-pose.pt')

video_path = 'videos/combat.mp4'

data_extractor = VideoDataExtractor(yolo_model)

video_information = data_extractor.get_video_information(video_path)

joblib.dump(video_information, 'combat-capture.sav')

vid_info = joblib.load('combat-capture.sav')

print(vid_info)

exit()

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

exit()

print('===========================')
print(np.mean(np.abs(frame_1[0] - frame_2[0])))
print('===========================')
print(np.mean(np.abs(frame_1[0] - frame_2[1])))

"""
fighter_red = frame_1[0]
fighter_blue = frame_1[1]

print(fighter_red)

print('================')
print(fighter_red)
print('================')
print(fighter_blue)
"""

exit()

for i, result in enumerate(results):
    print(f'{i} - {len(result)}')