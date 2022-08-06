import numpy as np
import math


def translate(ps):
    """
    :param ps: The points' positions
    :return: The new position
    """
    px, py, pz = ps
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [px, py, pz, 1]
    ])


def rotate_x(angle):
    """
    :param angle: The Angle of rotation about the X-axis
    :return: The rotation matrix about the x axis
    """
    return np.array([
        [1, 0, 0, 0],
        [0, math.cos(angle), math.sin(angle), 0],
        [0, -math.sin(angle), math.cos(angle), 0],
        [0, 0, 0, 1]
    ])


def rotate_y(angle):
    """
    :param angle: The Angle of rotation about the Y-axis
    :return: The rotation matrix about the y axis
    """
    return np.array([
        [math.cos(angle), 0, -math.sin(angle), 0],
        [0, 1, 0, 0],
        [math.sin(angle), 0, math.cos(angle), 0],
        [0, 0, 0, 1]
    ])


def rotate_z(angle):
    """
    :param angle: The Angle of rotation about the Z-axis
    :return: The rotation matrix about the z axis
    """
    return np.array([
        [math.cos(angle), math.sin(angle), 0, 0],
        [-math.sin(angle), math.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def scale(n):
    """
    :param n: n size
    :return: Scale change matrix
    """
    return np.array([
        [n, 0, 0, 0],
        [0, n, 0, 0],
        [0, 0, n, 0],
        [0, 0, 0, 1]
    ])