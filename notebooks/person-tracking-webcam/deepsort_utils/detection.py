import cv2
import numpy as np
from PIL import Image


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret



def xywh_to_xyxy(bbox_xywh, org_h, org_w):
    x, y, w, h = bbox_xywh
    x1 = max(int(x-w/2), 0)
    x2 = min(int(x+w/2), org_w-1)
    y1 = max(int(y-h/2), 0)
    y2 = min(int(y+h/2), org_h-1)
    return x1, y1, x2, y2


def xywh_to_tlwh(bbox_xywh):
    bbox_tlwh = bbox_xywh.copy()
    bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2]/2.
    bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3]/2.
    return bbox_tlwh


def tlwh_to_xyxy(bbox_tlwh, org_h, org_w):
    x, y, w, h = bbox_tlwh
    x1 = max(int(x), 0)
    x2 = min(int(x+w), org_w-1)
    y1 = max(int(y), 0)
    y2 = min(int(y+h), org_h-1)
    return x1, y1, x2, y2


def xyxy_to_tlwh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    t = x1
    l = y1
    w = int(x2-x1)
    h = int(y2-y1)
    return t, l, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
        cv2.putText(
            img,
            label,
            (x1, y1+t_size[1]+4),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            [255, 255, 255],
            2
        )
    return img