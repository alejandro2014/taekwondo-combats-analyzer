import joblib
"""
import random

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

video_info = load_video_info('output-combat3-20230911-210657.sav')
frames_number = get_frames_number(video_info)

frames_with_hit = [430, 447, 550, 1076, 1432, 2391, 6479, 7110]
frames_without_hit = get_frames_without_hit(frames_number, frames_with_hit)

ratio_train_test = 0.125

print(frames_number)

exit()
def flatten_fighters_list(frame):
    points = []

    for person in frame:
        for point in person:
            new_point = int(str(int(point[0] * 10000)) + str(int(point[1] * 10000)))

            points.append(new_point)

    return points

def get_indexes():
    length_video_frames = 9009

    all_frames = list(range(length_video_frames))

    hit_train_frames = [430, 447, 550, 1076, 1432, 2391, 6479]
    hit_test_frames = [7110]
    hit_frames = hit_train_frames + hit_test_frames

    ratio = len(hit_test_frames) / len(hit_frames)

    for i in hit_frames:
        all_frames.remove(i)

    random.shuffle(all_frames)

    nohit_frames = all_frames

    test_frames_number = int(len(nohit_frames) * ratio)

    nohit_test = nohit_frames[:test_frames_number]
    nohit_train = nohit_frames[test_frames_number:]

    return nohit_train, nohit_test, hit_train_frames, hit_test_frames

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
