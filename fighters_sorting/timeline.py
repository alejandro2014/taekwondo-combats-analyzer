import numpy as np

from person_queue import Queue

class Timeline:
    def __init__(self, persons_number=10):
        self.frame_number = 0
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
            return

        available_queues = list(range(len(self.persons_queues)))
        
        for person in persons:
            queue, available_queues = self.get_queue(person, available_queues)
            queue.append(person)

        for i in available_queues:
            self.persons_queues[i].append(None)

        print(self)

    def get_queue(self, person, available_queues):
        last_persons = self.get_last_persons(available_queues)
        deltas = [ self.calculate_delta(person, previous_person) for previous_person in last_persons ]
        chosen_delta = np.array(deltas).argmin()
        chosen_queue_index = available_queues[chosen_delta]
    
        del available_queues[chosen_queue_index]

        return self.persons_queues[chosen_queue_index], available_queues
    
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
    
    def __str__(self):
        str1 = 'TIMELINE\n'
        str1 += '========\n'

        for i, queue in enumerate(self.persons_queues):
            str1 += f'>> Queue {i} ({len(queue)})\n'
            str1 += '-------------\n'

            for person in queue.array:
                str1 += f'Person -> {person}\n'

        return str1
    
    def get_available_queues(self):
        return []
    
"""
available_queues = get_available_queues()

available_queues = [ 0, 1, 2, 3, 4, 5 ]

def insert_in_queue(queue, element):
    available_queues.remove(2)

print(available_queues)

insert_in_queue(2, None)

print(available_queues)


exit()
"""