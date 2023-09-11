import os

class FileLoader:
    def __init__(self):
        self.MODELS_DIR = 'weights'
        self.VIDEOS_DIR = 'videos'

    def get_available_videos(self):
        return self.get_available_files(self.VIDEOS_DIR)

    def get_available_models(self):
        return self.get_available_files(self.MODELS_DIR)

    def get_available_files(self, dir):
        return [ f'{dir}/{file}' for file in os.listdir(f'{dir}/') ]