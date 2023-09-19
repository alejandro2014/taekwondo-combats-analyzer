import joblib
import random

import numpy as np

class DatasetsFormatter:
    def __init__(self, frames_hit, ratio_test, input_combat_info_file):
        self.frames_hit = frames_hit
        self.ratio_test = ratio_test
        self.input_combat_info_file = input_combat_info_file

    def get_datasets(self, queue_number_1=0, queue_number_2=1):
        queues = self.load_video_info(self.input_combat_info_file)
        frames_number = len(queues[0])
        indices = self.get_train_and_test_indices(self.frames_hit, self.ratio_test, frames_number)

        # Colas de fotogramas detectadas como las que más detecciones no nulas comparten
        queue1 = queues[queue_number_1]
        queue2 = queues[queue_number_2]

        train = self.load_persons('train', queue1, queue2, indices)
        test = self.load_persons('test', queue1, queue2, indices)

        X_train = np.array([ self.flatten_fighters_list(list(e[0])) for e in train ])
        X_test = np.array([ self.flatten_fighters_list(list(e[0])) for e in test ])
        y_train = np.array([ e[1] for e in train ])
        y_test = np.array([ e[1] for e in test ])

        return X_train, X_test, y_train, y_test

    def load_video_info(self, info_path):
        return joblib.load(info_path)

    def get_frames_without_hit(self, frames_number, frames_with_hit):
        frames = list(range(frames_number))

        for i in frames_with_hit:
            frames.remove(i)

        return frames

    def list_splitter(self, frames, ratio):
        elements = len(frames)
        middle = int(elements * ratio)

        return [frames[:middle], frames[middle:]]

    def separate_frames(self, frames, ratio_test, hit_value):
        frames = [ (e, hit_value) for e in frames ]
        random.shuffle(frames)

        frames_test, frames_train = self.list_splitter(frames, ratio_test)

        return frames_test, frames_train

    def get_train_and_test_indices(self, frames_hit, ratio_test, frames_number):
        frames_nohit = self.get_frames_without_hit(frames_number, frames_hit)

        frames_hit_test, frames_hit_train = self.separate_frames(frames_hit, ratio_test, 1)
        frames_nohit_test, frames_nohit_train = self.separate_frames(frames_nohit, ratio_test, 0)

        frames_train = frames_nohit_train + frames_hit_train
        frames_test = frames_nohit_test + frames_hit_test

        print(f'frames_hit_train: {len(frames_hit_train)}')
        print(f'frames_hit_test: {len(frames_hit_test)}')
        print(f'frames_nohit_train: {len(frames_nohit_train)}')
        print(f'frames_nohit_test: {len(frames_nohit_test)}')
        print(f'frames_train: {len(frames_train)}')
        print(f'frames_test: {len(frames_test)}')

        random.shuffle(frames_train)
        random.shuffle(frames_test)

        return {
            'train': frames_train,
            'test': frames_test
        }

    def load_persons(self, dataset_type, queue1, queue2, indices):
        dataset = [
            ((queue1.array[i], queue2.array[i]), e[1])
            for i, e in enumerate(indices[dataset_type])
        ]

        print(f'Longitud del dataset original: {len(dataset)}')
        print(f'Longitud del dataset filtrado: {len([ p for p in dataset if p[0][0] is not None and p[0][1] is not None ])}')

        return [ p for p in dataset if p[0][0] is not None and p[0][1] is not None ]
        
    def flatten_fighters_list(self, frame):
        def int_to_str(float_num):
            numeric_part = str(int(float_num * 10000))
            zero_padding = '0' * (5 - len(numeric_part))

            return zero_padding + numeric_part

        points = []

        for person in frame:
            for point in person:
                points.append(
                    int(int_to_str(point[0]) + int_to_str(point[1]))
                )

        return points