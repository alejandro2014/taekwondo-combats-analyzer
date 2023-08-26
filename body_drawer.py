import cv2

class BodyDrawer:
    def draw(self, img, points):
        img = self.draw_head(img, points, (0, 255, 255))
        img = self.draw_trunk(img, points, (0, 255, 0))
        img = self.draw_arms(img, points, (255, 0, 0))
        img = self.draw_legs(img, points, (0, 0, 255))

        return img

    def draw_head(self, img, pose_points, color):
        points_2_draw = [ pose_points[joint] for joint in range(5) ]

        img = self.draw_points(img, points_2_draw, color)

        return img

    def draw_trunk(self, img, pose_points, color):
        lines = [(5, 6), (5, 11), (6, 12), (11, 12)]

        img = self.draw_lines(img, pose_points, lines, color)

        return img

    def draw_arms(self, img, pose_points, color):
        lines = [(5, 7), (7, 9), (6, 8), (8, 10)]

        points_2_draw = [ pose_points[joint] for joint in self.get_joints(lines) ]

        img = self.draw_points(img, points_2_draw, color)
        img = self.draw_lines(img, pose_points, lines, color)

        return img

    def draw_legs(self, img, pose_points, color):
        lines = [(11, 13), (13, 15), (12, 14), (14, 16)]

        points_2_draw = [ pose_points[joint] for joint in self.get_joints(lines) ]

        img = self.draw_points(img, points_2_draw, color)
        img = self.draw_lines(img, pose_points, lines, color)

        return img

    def draw_points(self, img, points, color):
        for point in points:
            img = cv2.circle(img, self.int_point(point), 3, color, 3)

        return img

    def draw_lines(self, img, points, lines, color):
        for line in lines:
            img = self.draw_line(img, points, line, color)

        return img
    
    def draw_line(self, img, points, line_points, color):
        start_point, end_point = line_points
        point1 = points[start_point]
        point2 = points[end_point]

        img = cv2.line(img, self.int_point(point1), self.int_point(point2), color, 2)
        
        return img

    def get_joints(self, lines):
        joints = list(sum(lines, ()))

        return list(set(joints))
    
    def int_point(self, point):
        return (int(point[0]), int(point[1]))