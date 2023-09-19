# Analizador de combates de taekwondo
Aquí se incluye la forma de ejecutar los módulos de la aplicación.
## 0. Instalación del software
Las librerías de software necesarias son las siguientes:
- backports.lzma
- joblib
- sklearn
- streamlit
- ultralytics

Para instalar los requisitos:
```
pip install -r requirements
```

## 1. Extracción de información de las personas (consola)
```
usage: python extract_video_results.py [-h] [--input-video INPUT_VIDEO] [--yolo-model YOLO_MODEL]

Extracts information of the provided video

options:
  -h, --help            show this help message and exit
  --input-video INPUT_VIDEO
                        Name of the video
  --yolo-model YOLO_MODEL
                        YOLO model to use
```

## 2. Extracción de información de las personas (GUI)
```
streamlit run extract_video_results_gui.py
```

## 3. Extracción y filtrado de información
```
cd fighters_sorting
python convert_coords.py {video_path}
```

## 4. Detección de golpes con el SVM
```
cd detect_hits
python detect_hits.py 
```
