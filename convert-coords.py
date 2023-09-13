import argparse
import joblib

import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(description='Sorts and filters the information of the fighters')

    parser.add_argument('input_data_file', help='Name of the data file of the video in .sav format')

    return parser.parse_args()

def load_video_info(info_path):
    return joblib.load(info_path)

def get_person_area(person):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

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

def get_person_delta(person1, person2):
    if len(person1) == 0 or len(person2) == 0:
        return 0
    
    return np.mean(np.abs(np.array(person1) - np.array(person2)))

def process_person_info(person, previous_frame):
    deltas = [ get_person_delta(person, previous_person) for previous_person in previous_frame ]
    person_area = get_person_area(person)

    return (person, person_area, deltas)

args = get_arguments()

vid_info = load_video_info(args.input_data_file)

input_frames = vid_info['results']#[1:]
output_frames = []

for i, frame in enumerate(input_frames):
    if i == 0:
        continue

    print(f'Processing frame {i}')

    persons_info = [ process_person_info(person, input_frames[i - 1]) for person in frame ]
    output_frames.append(persons_info)
        
print(output_frames)

exit()

"""
def calculate_delta(person1, person2):
    return np.mean(np.abs(person1 - person2))
    
def find_matches_previous_frame(person, candidates):
    deltas = [ calculate_delta(person, c) for c in candidates ]
    
    minimum = np.array(deltas).argmin()

    return person, minimum
"""

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