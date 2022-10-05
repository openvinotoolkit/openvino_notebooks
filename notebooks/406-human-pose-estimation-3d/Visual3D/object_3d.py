import cv2
import matrix_change as mc
import numpy as np


class Object3D:
    def __init__(self, render):
        self.render = render

        self.vertexes = np.array([])

        # np.array([
        # box
        # (0, 0, 0, 1),
        # (0, 1, 0, 1),
        # (1, 1, 0, 1),
        # (1, 0, 0, 1),
        # (0, 0, 1, 1),
        # (0, 1, 1, 1),
        # (1, 1, 1, 1),
        # (1, 0, 1, 1)

        # tetrahedron
        # (0, 0, 0, 1),
        # (0, 1, 0, 1),
        # (1, 1, 0, 1),
        # (1, 0, 1, 1)

        # ])

        self.faces = np.array(
            [
                # (0, 1, 2, 3),   # like these vertexes
                #                 # (0, 0, 0, 1),
                #                 # (0, 1, 0, 1),
                #                 # (1, 1, 0, 1),
                #                 # (1, 0, 0, 1),
                # (4, 5, 6, 7),
                # (0, 4, 5, 1),
                # (2, 3, 7, 6),
                # (1, 2, 6, 5),
                # (0, 3, 7, 4)]
                (0, 1, 2),
                (0, 2, 3),
                (1, 2, 3),
                (0, 1, 3),
            ]
        )

        # self.font = pg.font.SysFont('Arial', 30, bold=True)
        # self.color_faces = [(pg.Color('orange'), face) for face in self.faces]
        self.color_faces = [((255, 255, 255), face) for face in self.faces]
        self.Move_flag, self.draw_vertexes = True, True
        self.label = ""

    def translate(self, ps):
        self.vertexes = self.vertexes @ mc.translate(ps)

    def scale(self, size):
        self.vertexes = self.vertexes @ mc.scale(size)

    def rotate_x(self, angle):
        self.vertexes = self.vertexes @ mc.rotate_x(angle)

    def rotate_y(self, angle):
        self.vertexes = self.vertexes @ mc.rotate_y(angle)

    def rotate_z(self, angle):
        self.vertexes = self.vertexes @ mc.rotate_z(angle)

    def screen_projection(self):
        vertexes = self.vertexes @ self.render.camera.camera_matrix()
        vertexes = vertexes @ self.render.projection.projection_matrix
        vertexes /= vertexes[:, -1].reshape(-1, 1)
        vertexes[(vertexes > 2) | (vertexes < -2)] = 0
        vertexes = vertexes @ self.render.projection.to_screen_matrix
        vertexes = vertexes[:, :2]

        vertexes = vertexes.astype(int)
        # print(vertexes)

        # for face in self.faces:
        for index, color_face in enumerate(self.color_faces):
            color, face = color_face
            polygon = vertexes[face]

            if not np.any(
                (polygon == self.render.H_WIDTH) | (polygon == self.render.H_HEIGHT)
            ):

                if polygon.shape[0] == 2:
                    cv2.line(
                        self.render.screen,
                        polygon[0],
                        polygon[1],
                        color,
                        3,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.polylines(self.render.screen, [polygon], True, color)

        if self.draw_vertexes:
            for vertex in vertexes:
                if not np.any(
                    (vertex == self.render.H_WIDTH) | (vertex == self.render.H_HEIGHT)
                ):
                    cv2.circle(self.render.screen, vertex, 6, color, -1)

    def draw(self):
        self.screen_projection()
        # self.Move()

    # def Move(self):
    #     if self.Move_flag:
    #         # self.rotate_x(pg.time.get_ticks() % 0.005)
    #         self.rotate_x(cv2.getTickCount() % 0.005)


class Axes(Object3D):
    def __init__(self, render):
        super().__init__(render)
        self.vertexes = np.array(
            [(0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        )
        self.faces = np.array([(0, 1), (0, 2), (0, 3)])
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.color_faces = [
            (color, face) for color, face in zip(self.colors, self.faces)
        ]
        self.draw_vertexes = False
        # self.label = 'XYZ'


class Grid(Object3D):
    def __init__(self, render, grid_size, dimension):
        super().__init__(render)

        grid_array = []
        step = grid_size / dimension
        dimension = dimension // 2
        for step_id in range(dimension + 1):  # add grid
            # grid_array.append(np.array([[-grid_size / 2, -grid_size / 2 + step_id * step, 0, 1],
            #                             [grid_size / 2, -grid_size / 2 + step_id * step, 0, 1]], dtype=np.float32))
            # grid_array.append(np.array([[-grid_size / 2 + step_id * step, -grid_size / 2, 0, 1],
            #                             [-grid_size / 2 + step_id * step, grid_size / 2, 0, 1]], dtype=np.float32))
            # grid_array.append(np.array([[-grid_size / 2, 0, -grid_size / 2 + step_id * step, 1],
            #                             [grid_size / 2, 0, -grid_size / 2 + step_id * step, 1]], dtype=np.float32))
            # grid_array.append(np.array([[-grid_size / 2 + step_id * step, 0, -grid_size / 2, 1],
            #                             [-grid_size / 2 + step_id * step, 0, grid_size / 2, 1]], dtype=np.float32))

            # divide into 4 blocks

            # grid_array.append(np.array([-grid_size / 2, 0, -grid_size / 2 + step_id * step, 1], dtype=np.float32))
            # grid_array.append(np.array([grid_size / 2, 0, -grid_size / 2 + step_id * step, 1], dtype=np.float32))
            # grid_array.append(np.array([-grid_size / 2 + step_id * step, 0, -grid_size / 2, 1], dtype=np.float32))
            # grid_array.append(np.array([-grid_size / 2 + step_id * step, 0, grid_size / 2, 1], dtype=np.float32))

            grid_array.append(np.array([0, 0, 0 + step_id * step, 1], dtype=np.float32))
            grid_array.append(
                np.array([grid_size / 2, 0, 0 + step_id * step, 1], dtype=np.float32)
            )
            grid_array.append(np.array([0 + step_id * step, 0, 0, 1], dtype=np.float32))
            grid_array.append(
                np.array([0 + step_id * step, 0, grid_size / 2, 1], dtype=np.float32)
            )

        self.vertexes = np.array(grid_array)
        # print(self.vertexes)

    def screen_projection(self):
        vertexes = self.vertexes @ self.render.camera.camera_matrix()
        vertexes = vertexes @ self.render.projection.projection_matrix
        vertexes /= vertexes[:, -1].reshape(-1, 1)
        # vertexes[(vertexes > 2) | (vertexes < -2)] = 0
        vertexes = vertexes @ self.render.projection.to_screen_matrix
        vertexes = vertexes[:, :2]

        vertexes = vertexes.astype(int)
        vertexes = vertexes.reshape(-1, 2, 2)
        # print(f'vertexes is {vertexes}')
        
        for grid_line in vertexes:
            if not np.any(
                (grid_line == self.render.H_WIDTH) | (grid_line == self.render.H_HEIGHT)
            ):
                # print(grid_line[0], grid_line[1])
                cv2.line(
                    self.render.screen,
                    tuple(grid_line[0]),
                    tuple(grid_line[1]),
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )


class Skeleton(Object3D):
    def __init__(self, render):
        super().__init__(render)
        # self.vertexes = []
        # self.body_edge = []

    def set_body(self, joints, body_edge, size=1.5):
        self.vertexes = joints
        self.body_edge = body_edge
        # change the scale of skeleton
        self.vertexes = self.vertexes @ mc.scale(size)

    def screen_projection(self):
        if len(self.vertexes) < 1:
            return 0
        for ver in self.vertexes:
            vertexes = ver @ self.render.camera.camera_matrix()
            vertexes = vertexes @ self.render.projection.projection_matrix
            vertexes /= vertexes[:, -1].reshape(-1, 1)
            vertexes[(vertexes > 2) | (vertexes < -2)] = 0
            vertexes = vertexes @ self.render.projection.to_screen_matrix
            vertexes = vertexes[:, :2]

            vertexes = vertexes.astype(int)
            # vertexes = vertexes.reshape(-1, 2, 2)
            # print(f'vertexes is {vertexes[:2]}')

            for edge in self.body_edge:
                if not np.any(
                    (vertexes[edge[0]] == self.render.H_WIDTH)
                    | (vertexes[edge[1]] == self.render.H_HEIGHT)
                ):
                    # print(grid_line[0], grid_line[1])
                    cv2.line(
                        self.render.screen,
                        tuple(vertexes[edge[0]]),
                        tuple(vertexes[edge[1]]),
                        (0, 100, 255),
                        2,
                        cv2.LINE_AA,
                    )

        self.vertexes = []