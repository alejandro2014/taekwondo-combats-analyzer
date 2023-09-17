import joblib
import random

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def load_video_info(info_path):
    return joblib.load(info_path)

def get_frames_number(video_info):
    return len(video_info[0])

def get_frames_without_hit(frames_number, frames_with_hit):
    frames = list(range(frames_number))

    for i in frames_with_hit:
        frames.remove(i)

    return frames

def list_splitter(frames, ratio):
    elements = len(frames)
    middle = int(elements * ratio)

    return [frames[:middle], frames[middle:]]

def separate_frames(frames, ratio_test, hit_value):
    frames = [ (e, hit_value) for e in frames ]
    random.shuffle(frames)

    frames_train, frames_test = list_splitter(frames, ratio_test)

    return frames_train, frames_test

def get_train_and_test_indices(frames_hit, ratio_test, frames_number):
    frames_nohit = get_frames_without_hit(frames_number, frames_hit)

    frames_hit_train, frames_hit_test = separate_frames(frames_hit, ratio_test, 1)
    frames_nohit_train, frames_nohit_test = separate_frames(frames_nohit, ratio_test, 0)

    frames_train = frames_nohit_train + frames_hit_train
    frames_test = frames_nohit_test + frames_hit_test

    random.shuffle(frames_train)
    random.shuffle(frames_test)

    return {
        'train': frames_train,
        'test': frames_test
    }

def load_persons(dataset_type, queue1, queue2, indices):
    dataset = [
        (
            ( queue1.array[i], queue2.array[i] ),
            e[1]
        ) for i, e in enumerate(indices[dataset_type])
    ]

    return [ p for p in dataset if p[0][0] is not None and p[0][1] is not None ]
    
def get_datasets(frames_hit, ratio_test, input_combat_info_file):
    queues = load_video_info(input_combat_info_file)
    frames_number = get_frames_number(queues)
    indices = get_train_and_test_indices(frames_hit, ratio_test, frames_number)

    # Colas de fotogramas detectadas como las que más detecciones no nulas comparten
    queue1 = queues[1]
    queue2 = queues[2]

    train = load_persons('train', queue1, queue2, indices)
    test = load_persons('test', queue1, queue2, indices)

    return train, test

frames_hit = [ 430, 447, 550, 1076, 1432, 2391, 6479, 7110 ]
ratio_test = 0.25
input_combat_info_file = 'output-combat3-20230911-210657.sav'

train_dataset, test_dataset = get_datasets(frames_hit, ratio_test, input_combat_info_file)

print(train_dataset[:10])
exit()
#--------------------------------------------------------------------------
#print([ e[0] for e in train_dataset ])

X_train = np.array([ e[0] for e in train_dataset ])
X_test = np.array([ e[0] for e in test_dataset ])
y_train = np.array([ e[1] for e in train_dataset ])
y_test = np.array([ e[1] for e in test_dataset ])

svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# Luego, puedes usar el modelo entrenado para predecir golpes en nuevas listas de pose.
# Supongamos que tienes una nueva lista de pose en forma de un array llamado 'nueva_lista_pose'.
nueva_lista_pose = np.array([1.2, 2.3, 0.5, ...])  # Reemplaza con valores reales de tu lista de pose

# Realiza una predicción de golpe
prediccion_golpe = svm_model.predict([nueva_lista_pose])

# La variable 'prediccion_golpe' ahora contiene 1 si se predice un golpe, o 0 si no se predice un golpe.

exit()
#--------------------------------------------------------------------------
X = nohit_train + hit_train_frames + nohit_test + hit_test_frames
y = [0] * len(nohit_train) + [1] * len(hit_train_frames) + [0] * len(nohit_test) + [1] * len(hit_test_frames)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio_test, random_state=52)

flat_results = [ flatten_fighters_list(frame) for frame in video_info['results'] ]

nohit_train, nohit_test, hit_train_frames, hit_test_frames = get_indexes()

frames = video_info['results']

nohit_train = get_persons_by_indices(frames, nohit_train)
hit_train_frames = get_persons_by_indices(frames, hit_train_frames)

nohit_test = get_persons_by_indices(frames, nohit_test)
hit_test_frames = get_persons_by_indices(frames, hit_test_frames)

def get_metrics(dataset):
    not_null = [ p for p in dataset if p[0][0] is not None and p[0][1] is not None ]

    return {
        'not_null': not_null,
        'not_null_hit': [ p for p in not_null if p[1] == 1 ],
        'not_null_nohit': [ p for p in not_null if p[1] == 0 ]
    }

train_metrics = get_metrics(train)
test_metrics = get_metrics(test)

print(f"train: {len(train)}")
print(f"train_not_null: {len(train_metrics['not_null'])}")
print(f"train_not_null_hit: {len(train_metrics['not_null_hit'])}")
print(f"train_not_null_nohit: {len(train_metrics['not_null_nohit'])}")
print('-----------------------')
print(f"test: {len(test)}")
print(f"test_not_null: {len(test_metrics['not_null'])}")
print(f"test_not_null_hit: {len(test_metrics['not_null_hit'])}")
print(f"test_not_null_nohit: {len(test_metrics['not_null_nohit'])}")

exit()

def show_filled_persons(queues, indices):
    queues_no = len(queues)

    for i in range(queues_no):
        for j in range(queues_no):
            persons = load_persons(queues, indices, i, j)
            print(len([ p for p in persons if p[0][0] is not None and p[0][1] is not None ]), end='')
            print(' ', end='')
        print()

def flatten_fighters_list(frame):
    points = []

    for person in frame:
        for point in person:
            new_point = int(str(int(point[0] * 10000)) + str(int(point[1] * 10000)))

            points.append(new_point)

    return points

def get_persons_by_indices(frames, indices):
    return [ flatten_fighters_list(frames[i]) for i in indices ]