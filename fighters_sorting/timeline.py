import joblib
import numpy as np

from person_queue import Queue

class Timeline:
    def __init__(self, persons_number=10):
        self.persons_number = persons_number
        self.deltas = [ None ] * persons_number
        self.persons_queues = [ ]

        for _ in range(persons_number):
            self.persons_queues.append(Queue())

    def insert_frames(self, frames):
        for frame in frames:
            self.insert_persons(frame)

    def insert_persons(self, persons):
        if not persons[0]:
            for queue in self.persons_queues:
                queue.append(None)

            return

        available_queues = self.insert_normal_persons(persons)

        self.insert_none_persons(available_queues)

    def insert_normal_persons(self, persons):
        available_queues = list(range(len(self.persons_queues)))

        for person in persons:
            queue_position = self.get_queue_number(person, available_queues)
            available_queues = self.insert_person(queue_position, person, available_queues)
        
        return available_queues

    def insert_none_persons(self, available_queues):
        for i in available_queues:
            self.persons_queues[i].append(None)

    def insert_person(self, queue_position, person, available_queues):
        queue = self.persons_queues[queue_position]
        queue.append(person)
        available_queues.remove(queue_position)

        return available_queues

    def get_queue_number(self, person, available_queues):
        previous_persons = self.get_last_persons(available_queues)
        deltas = [ self.calculate_delta(person, previous_person) for previous_person in previous_persons ]

        min_delta_position = np.array(deltas).argmin()

        return available_queues[min_delta_position]
    
    def get_last_persons(self, available_queues):
        queues = [ self.persons_queues[i] for i in available_queues ]

        return [ self.get_person_from_queue(queue) for queue in queues ]
    
    def get_person_from_queue(self, queue):
        if not queue or all(e is None for e in queue.array):
            return None

        return [ e for e in queue.array if e is not None ][-1]
    
    def calculate_delta(self, person1, person2):
        if person1 is None or person2 is None or len(person1) == 0 or len(person2) == 0:
            return 1000
        
        return np.mean(np.abs(np.array(person1) - np.array(person2)))
    
    def write_to_file(self, output_data_file):
        joblib.dump(self.persons_queues, output_data_file)

    def __str__(self):
        str1 = 'TIMELINE\n'
        str1 += '========\n'

        for i, queue in enumerate(self.persons_queues):
            str1 += f'>> Queue {i} ({len(queue)})\n'
            str1 += '-------------\n'

            for person in queue.array:
                str1 += f'Person -> {person}\n'

        return str1