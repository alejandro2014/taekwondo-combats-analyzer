import joblib
import random

"""
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
"""

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

def get_train_and_test_indices(frames_hit, ratio_test):
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

def load_persons(queues, indices, person1_track, person2_track):
    return [
        ((
            queues[person1_track].array[i],
            queues[person2_track].array[i]
        ), e[1])
        for i, e in enumerate(indices['train'])
    ]
    
frames_hit = [ 430, 447, 550, 1076, 1432, 2391, 6479, 7110 ]
ratio_test = 0.25
input_combat_info_file = 'output-combat3-20230911-210657.sav'
queue_person1 = 0
queue_person2 = 1

queues = load_video_info(input_combat_info_file)
frames_number = get_frames_number(queues)

indices = get_train_and_test_indices(frames_hit, ratio_test)

print(f'q{queue_person1}, q{queue_person2} -> ', end='')
persons = load_persons(queues, indices, queue_person1, queue_person2)
print(len([ p for p in persons if p[0][0] is not None and p[0][1] is not None ]))

exit()

def flatten_fighters_list(frame):
    points = []

    for person in frame:
        for point in person:
            new_point = int(str(int(point[0] * 10000)) + str(int(point[1] * 10000)))

            points.append(new_point)

    return points

def get_persons_by_indices(frames, indices):
    return [ flatten_fighters_list(frames[i]) for i in indices ]

video_info = load_video_info('combat3-20230911-210657.sav')

flat_results = [ flatten_fighters_list(frame) for frame in video_info['results'] ]

nohit_train, nohit_test, hit_train_frames, hit_test_frames = get_indexes()

frames = video_info['results']

nohit_train = get_persons_by_indices(frames, nohit_train)
hit_train_frames = get_persons_by_indices(frames, hit_train_frames)

nohit_test = get_persons_by_indices(frames, nohit_test)
hit_test_frames = get_persons_by_indices(frames, hit_test_frames)

X = nohit_train + hit_train_frames + nohit_test + hit_test_frames
y = [0] * len(nohit_train) + [1] * len(hit_train_frames) + [0] * len(nohit_test) + [1] * len(hit_test_frames)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=52)

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
