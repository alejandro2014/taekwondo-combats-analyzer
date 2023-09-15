class Person:
    def __init__(self, points):
        self.points = points
        self.size = self.calculate_size()

    def calculate_size(self):
        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0

        for point in self.points:
            if point[0] < min_x:
                min_x = point[0]

            if point[0] > max_x:
                max_x = point[0]

            if point[1] < min_y:
                min_y = point[1]

            if point[1] > max_y:
                max_y = point[1]

        return abs(max_x - min_x) * abs(max_y - min_y)

    def __str__(self):
        return f'Person\n------\nsize: {self.size}\n{self.points}'