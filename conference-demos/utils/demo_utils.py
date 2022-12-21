import os.path
import sys

import cv2

sys.path.append("../../notebooks/utils")
from notebook_utils import *

logo_img = cv2.imread(os.path.join(os.path.dirname(__file__), "openvino-logo.png"), cv2.IMREAD_UNCHANGED)
def draw_ov_watermark(frame, alpha=0.35, size=0.15):
    scale = size * frame.shape[1] / logo_img.shape[1]
    watermark = cv2.resize(logo_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    alpha_channel = watermark[:, :, 3:].astype(float) / 255
    alpha_channel *= alpha
    patch = frame[frame.shape[0] - watermark.shape[0]:, frame.shape[1] - watermark.shape[1]:]

    patch[:] = alpha_channel * watermark[:, :, :3] + ((1.0 - alpha_channel) * patch)