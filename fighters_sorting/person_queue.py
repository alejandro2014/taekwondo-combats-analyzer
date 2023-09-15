class Queue:
    def __init__(self):
        self.array = [ ]

    def is_empty(self):
        return (not self.array or all(e is None for e in self.array))

    def value(self):
        if self.is_empty():
            return None
        
        return [ e for e in self.array if e is not None ][-1]
    
    def append(self, person):
        self.array.append(person)

    def __str__(self):
        str1 = 'Queue\n'
        str1 += '------------'

        for person in self.array:
            str1 += str(person)
            str1 += '\n'

        return str1
    
    def __len__(self):
        return len(self.array)