from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from datasets_formatter import DatasetsFormatter

frames_hit = [ 430, 447, 550, 1076, 1432, 2391, 6479, 7110 ]
ratio_test = 0.25
input_combat_info_file = 'output-combat3-20230911-210657.sav'

datasets_formatter = DatasetsFormatter(frames_hit, ratio_test, input_combat_info_file)
X_train, X_test, y_train, y_test = datasets_formatter.get_datasets(queue_number_1 = 1, queue_number_2 = 2)

model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)