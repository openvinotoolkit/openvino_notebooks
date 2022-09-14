import math
import numpy as np


class Projection:
    def __init__(self, render):
        NEAR_JP = render.camera.near_plane
        FAR_JP = render.camera.far_plane
        RIGHT_JP = math.tan(render.camera.h_fov / 2)
        LEFT_JP = -RIGHT_JP
        TOP_JP = math.tan(render.camera.v_fov / 2)
        BOTTOM_JP = -TOP_JP

        m00 = 2 / (RIGHT_JP - LEFT_JP)
        m11 = 2 / (TOP_JP - BOTTOM_JP)
        m22 = (FAR_JP + NEAR_JP) / (FAR_JP - NEAR_JP)
        m32 = -2 * NEAR_JP * FAR_JP / (FAR_JP - NEAR_JP)
        self.projection_matrix = np.array([
            [m00, 0, 0, 0],
            [0, m11, 0, 0],
            [0, 0, m22, 1],
            [0, 0, m32, 0]
        ])

        W, H = render.H_WIDTH, render.H_HEIGHT
        self.to_screen_matrix = np.array([
            [W, 0, 0, 0],
            [0, -H, 0, 0],
            [0, 0, 1, 0],
            [W, H, 0, 1]
        ])

    def get_projection_matrix(self):
        return self.projection_matrix

    def get_to_screen_matrix(self):
        return self.to_screen_matrix
