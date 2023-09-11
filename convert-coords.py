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

person = [
        [0.011168761178851128, 0.42047256231307983],
        [0.011408274993300438, 0.4150969684123993],
        [0.008551133796572685, 0.41479504108428955],
        [0.008481677621603012, 0.4190683662891388],
        [0.004170811269432306, 0.4181576669216156],
        [0.011019224300980568, 0.44985875487327576],
        [0.006610413547605276, 0.45065930485725403],
        [0.021875737234950066, 0.4937007427215576],
        [0.018324116244912148, 0.4970078766345978],
        [0.025260956957936287, 0.5108044743537903],
        [0.025708375498652458, 0.510515570640564],
        [0.01083137933164835, 0.5288711190223694],
        [0.0092965979129076, 0.5292530655860901],
        [0.018044190481305122, 0.5325304865837097],
        [0.015144234523177147, 0.5328734517097473],
        [0.01911449246108532, 0.5494314432144165],
        [0.01832694001495838, 0.5501647591590881]
    ]

print(calculate_person_area(person))

exit()
import argparse
import joblib

def get_arguments():
    parser = argparse.ArgumentParser(description='Sorts and filters the information of the fighters')

    parser.add_argument('input_data_file', help='Name of the data file of the video in .sav format')

    return parser.parse_args()

def load_video_info(info_path):
    return joblib.load(info_path)

args = get_arguments()

vid_info = load_video_info(args.input_data_file)

print(vid_info['path'])
print(vid_info['properties'])

for i, result in enumerate(vid_info['results'][:100]):
    print('-------------------------------')
    print(f'Frame {i}')

    for person in result:
        print('person')

exit()

sequence = [[
    [
        [0.011168761178851128, 0.42047256231307983],
        [0.011408274993300438, 0.4150969684123993],
        [0.008551133796572685, 0.41479504108428955],
        [0.008481677621603012, 0.4190683662891388],
        [0.004170811269432306, 0.4181576669216156],
        [0.011019224300980568, 0.44985875487327576],
        [0.006610413547605276, 0.45065930485725403],
        [0.021875737234950066, 0.4937007427215576],
        [0.018324116244912148, 0.4970078766345978],
        [0.025260956957936287, 0.5108044743537903],
        [0.025708375498652458, 0.510515570640564],
        [0.01083137933164835, 0.5288711190223694],
        [0.0092965979129076, 0.5292530655860901],
        [0.018044190481305122, 0.5325304865837097],
        [0.015144234523177147, 0.5328734517097473],
        [0.01911449246108532, 0.5494314432144165],
        [0.01832694001495838, 0.5501647591590881]
    ],
    [
        [0.4089016318321228, 0.3582136631011963],
        [0.4097466766834259, 0.3519134521484375],
        [0.40981003642082214, 0.34657996892929077],
        [0.37695616483688354, 0.33515962958335876],
        [0.40184301137924194, 0.3274875283241272],
        [0.31930747628211975, 0.35255712270736694],
        [0.41240057349205017, 0.34312769770622253],
        [0.2877998948097229, 0.40527084469795227],
        [0.4554271996021271, 0.3615698218345642],
        [0.32225102186203003, 0.42474061250686646],
        [0.4634215831756592, 0.3501257598400116],
        [0.32339057326316833, 0.48002341389656067],
        [0.4042518734931946, 0.48174580931663513],
        [0.3141513764858246, 0.5888193845748901],
        [0.5067970156669617, 0.5844878554344177],
        [0.2714751064777374, 0.6567552089691162],
        [0.5886415243148804, 0.6715880632400513]
    ]
]]

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