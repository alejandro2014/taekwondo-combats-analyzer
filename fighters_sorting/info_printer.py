class FramesInfoPrinter:
    def show_frames_info(self, frames):
        frames_number = len(frames)
        histogram = {}

        print(f'Frames: {frames_number}')

        print('Sequences:')
        for frame in frames:
            persons_detected = len(frame)
            print(f'{persons_detected}', end='')
        
        print()

        print('Histogram:')
        histogram = self.calculate_histogram(frames)
        print(histogram)
        
    def calculate_histogram(self, frames):
        dict1 = {}

        for frame in frames:
            persons_detected = len(frame)

            if persons_detected not in dict1:
                dict1[persons_detected] = 1
            else:
                dict1[persons_detected] += 1

        new_dict = {}
        total = sum(dict1.values())

        for k in list(dict1.keys()):
            new_dict[k] = (dict1[k], round(dict1[k] / total * 100, 3))

        return new_dict