import argparse
import joblib

import numpy as np

[[(person, area, previous)]] = None

def get_arguments():
    parser = argparse.ArgumentParser(description='Sorts and filters the information of the fighters')

    parser.add_argument('input_data_file', help='Name of the data file of the video in .sav format')

    return parser.parse_args()

def load_video_info(info_path):
    return joblib.load(info_path)

def calculate_person_area(person):
    min_x = 1
    max_x = -1
    min_y = 1
    max_y = -1

    for point in person:
        if point[0] < min_x:
            min_x = point[0]

        if point[0] > max_x:
            max_x = point[0]

        if point[1] < min_y:
            min_y = point[1]

        if point[1] > max_y:
            max_y = point[1]

    return abs(max_x - min_x) * abs(max_y - min_y)

def calculate_delta(person1, person2):
    return np.mean(np.abs(person1 - person2))
    
def find_matches_previous_frame(person, candidates):
    deltas = [ calculate_delta(person, c) for c in candidates ]
    
    minimum = np.array(deltas).argmin()

    return person, minimum
        

args = get_arguments()

vid_info = load_video_info(args.input_data_file)

print(vid_info['path'])
print(vid_info['properties'])

frames_info = []

for i, result in enumerate(vid_info['results']):
    print('-------------------------------')
    print(f'Frame {i}')

    frames_info.append([])

    for person in result:
        print(calculate_person_area(person))

        if i > 0:
            find_matches_previous_frame(person, result[i - 1])
        else:
            frames_info.append(person)

exit()

def convert_point(point):
    coord1_transf = int(point[0] * 1000) * 10000
    coord2_transf = int(point[1] * 1000) + 1000

    return coord1_transf + coord2_transf

def process_fighter(fighter):
    return [ convert_point(point) for point in fighter ]

def process_frame(frame):
    return process_fighter(frame[0]) + process_fighter(frame[1])

def process_sequence(sequence):
    return [ process_frame(frame) for frame in sequence ]

list_values = process_sequence(sequence)

print(list_values)

exit()

import joblib
import numpy as np

frames = joblib.load('combat-capture.sav')

print(len(frames))
exit()

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