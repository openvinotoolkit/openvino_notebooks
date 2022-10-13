import numpy as np
import cv2, sys
import matrix_change as mc
import math


class Camera:
    """
    define the camera and it's attribute
    """
    def __init__(self, render, position):
        self.render = render
        self.position = np.array([*position, 1.0])

        self.right = np.array([1, 0, 0, 1])
        self.up = np.array([0, 1, 0, 1])
        self.forward = np.array([0, 0, 1, 1])

        self.h_fov = math.pi / 2
        self.v_fov = self.h_fov * (render.HEIGHT / render.WIDTH)
        self.near_plane = 0.01
        self.far_plane = 100

        # set moving profile
        self.moving_speed = 0.2
        self.rotation_speed = 0.1
        
        # turn left 0.1
        self.camera_yaw(-self.rotation_speed)

    """
    Press and hold on these follow buttons to move the camera.
    """
    def control(self):
        key = cv2.waitKey(1)
        # print(key)
        if key == ord('a'):
            self.position -= self.right * self.moving_speed
        if key == ord('d'):
            self.position += self.right * self.moving_speed
        if key == ord('w'):
            self.position += self.forward * self.moving_speed
        if key == ord('s'):
            self.position -= self.forward * self.moving_speed
        if key == ord('q'):
            self.position += self.up * self.moving_speed
        if key == ord('e'):
            self.position -= self.up * self.moving_speed
        # move camera
        if key == ord('j'):
            self.camera_yaw(-self.rotation_speed)
        if key == ord('l'):
            self.camera_yaw(self.rotation_speed)
        if key == ord('i'):
            self.camera_pitch(-self.rotation_speed)
        if key == ord('k'):
            self.camera_pitch(self.rotation_speed)
        
        if key == 27:
            sys.exit()

    # translate
    def translate_matrix(self):
        x, y, z, w = self.position
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [-x, -y, -z, 1]
        ])

    # rotate
    def rotate_matrix(self):
        rx, ry, rz, w = self.right
        fx, fy, fz, w = self.forward
        ux, uy, uz, w = self.up
        return np.array([
            [rx, ux, fx, 0],
            [ry, uy, fy, 0],
            [rz, uz, fz, 0],
            [0, 0, 0, 1]
        ])

    # camera view
    def camera_matrix(self):
        return self.translate_matrix() @ self.rotate_matrix()

    def camera_yaw(self, angle):
        rotate = mc.rotate_y(angle)
        self.forward = self.forward @ rotate
        self.right = self.right @ rotate
        self.up = self.up @ rotate

    def camera_pitch(self, angle):
        rotate = mc.rotate_x(angle)
        self.forward = self.forward @ rotate
        self.right = self.right @ rotate
        self.up = self.up @ rotate