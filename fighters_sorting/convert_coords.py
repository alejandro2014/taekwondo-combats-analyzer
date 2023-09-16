import argparse
import joblib

import numpy as np

from info_printer import FramesInfoPrinter
from timeline import Timeline

def get_arguments():
    parser = argparse.ArgumentParser(description='Sorts and filters the information of the fighters')

    parser.add_argument('input_data_file', help='Name of the data file of the video in .sav format')

    return parser.parse_args()

def load_video_info(info_path):
    return joblib.load(info_path)

args = get_arguments()
vid_info = load_video_info(args.input_data_file)
frames = vid_info['results']

#info_printer = FramesInfoPrinter()
#info_printer.show_frames_info(frames)

timeline = Timeline(persons_number=10)
timeline.insert_frames(frames)

output_data_file = f'output-{args.input_data_file}'
timeline.write_to_file(output_data_file)

exit()

def calculate_delta(person1, person2):
    if person1 is None or person2 is None:
        return 0

    return round(np.mean(np.abs(np.array(person1) - np.array(person2))), 5)

def show_queues(queue):
    elements = [ e for e in queue.array ]

    for j in range(len(elements)):
        if j == 0:
            continue

        delta = calculate_delta(elements[j], elements[j - 1])

        print(delta)