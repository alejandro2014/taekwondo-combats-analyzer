import argparse
import joblib

import numpy as np

from timeline import Timeline

def get_arguments():
    parser = argparse.ArgumentParser(description='Sorts and filters the information of the fighters')

    parser.add_argument('input_data_file', help='Name of the data file of the video in .sav format')

    return parser.parse_args()

def load_video_info(info_path):
    return joblib.load(info_path)

def get_person_delta(person1, person2):
    if len(person1) == 0 or len(person2) == 0:
        return 0
    
    return np.mean(np.abs(np.array(person1) - np.array(person2)))

def process_person_info(person, previous_frame):
    deltas = [ get_person_delta(person, previous_person) for previous_person in previous_frame ]
    
    return (person, deltas)

def add_persons_information(input_frames):
    output_frames = []

    for i, frame in enumerate(input_frames):
        if i == 0:
            continue

        persons_info = [ process_person_info(person, input_frames[i - 1]) for person in frame ]
        output_frames.append(persons_info)

    return output_frames

def sort_persons_in_frames(input_frames):
    output_frames = []

    for frame in input_frames:
        output_frame = []

        for person in frame:
            person_keypoints, deltas = person
            
            chosen_delta = np.array(deltas).argmin()

            output_frame.append((person_keypoints, deltas, chosen_delta))

        output_frames.append(output_frame)

    return output_frames
    
args = get_arguments()
vid_info = load_video_info(args.input_data_file)
frames = vid_info['results'][:100]

timeline = Timeline(persons_number=5)

print(timeline)
timeline.insert_frames(frames)

exit()

args = get_arguments()

vid_info = load_video_info(args.input_data_file)
frames = vid_info['results']

current_frame = 0

frames = add_persons_information(frames)
frames = sort_persons_in_frames(frames)

matrix = compose_matrix(frames)

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