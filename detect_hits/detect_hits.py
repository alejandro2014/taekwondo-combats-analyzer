import random
import joblib
"""
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
"""

def load_video_info(info_path):
    return joblib.load(info_path)

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

nohit_train, nohit_test, hit_train_frames, hit_test_frames = get_indexes()

#video_info = load_video_info('./output-combat3-20230911-210657.sav')

exit()
# Supongamos que tienes un conjunto de datos etiquetado con listas de pose y etiquetas de golpe (0 o 1).
# X_train es una matriz de listas de pose y y_train es un array de etiquetas.

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrena un modelo de Máquinas de Vectores de Soporte (SVM)
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = svm_model.predict(X_test)

# Evalúa el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# Luego, puedes usar el modelo entrenado para predecir golpes en nuevas listas de pose.
# Supongamos que tienes una nueva lista de pose en forma de un array llamado 'nueva_lista_pose'.
nueva_lista_pose = np.array([1.2, 2.3, 0.5, ...])  # Reemplaza con valores reales de tu lista de pose

# Realiza una predicción de golpe
prediccion_golpe = svm_model.predict([nueva_lista_pose])

# La variable 'prediccion_golpe' ahora contiene 1 si se predice un golpe, o 0 si no se predice un golpe.
