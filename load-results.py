import cv2

def get_properties(video_path):
    properties_names = [
        'FPS',
        'FRAME_COUNT',
        'FRAME_HEIGHT',
        'FRAME_WIDTH'
    ]

    vid_cap = cv2.VideoCapture(video_path)

    return {
        property: vid_cap.get(getattr(cv2, f'CAP_PROP_{property}'))
        for property in properties_names
    }

properties = get_properties('videos/combat.mp4')
print(properties)

exit()

for property in properties:
    property_name = f'CAP_PROP_{property}'
    property_value = vid_cap.get(getattr(cv2, property_name))

    print(f'{property_name} = {property_value}')

exit()
import ffmpeg

media_file = './videos/combat.mp4'

print(dir(ffmpeg))
adios = ffmpeg.probe(media_file)
hola = adios["streams"]

print(hola)

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