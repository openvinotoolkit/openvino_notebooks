import math
from pathlib import Path

import cv2
import numpy as np
import pyclipper
import yaml
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon


def _read_yaml(config_path):
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _transform_op(config, name):
    for op in config["PreProcess"]["transform_ops"]:
        if name in op:
            return op[name] or {}
    return {}


def load_det_config(config_path):
    """
    Read text detection pre- and post-processing parameters from the model's inference.yml.

    `DetResizeForTest` is empty in the PP-OCRv6 inference.yml, so its PaddleOCR defaults are used
    (limit_side_len=736, limit_type="min", max_side_limit=4000).
    """
    config = _read_yaml(config_path)
    normalize = _transform_op(config, "NormalizeImage")
    resize = _transform_op(config, "DetResizeForTest")
    scale = normalize.get("scale", 1.0 / 255.0)
    if isinstance(scale, str):
        numerator, denominator = scale.split("/")
        scale = float(numerator) / float(denominator)
    post = config["PostProcess"]
    return {
        "mean": np.array(normalize.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32),
        "std": np.array(normalize.get("std", [0.229, 0.224, 0.225]), dtype=np.float32),
        "scale": float(scale),
        "limit_side_len": resize.get("limit_side_len", 736),
        "limit_type": resize.get("limit_type", "min"),
        "max_side_limit": resize.get("max_side_limit", 4000),
        "thresh": post.get("thresh", 0.3),
        "box_thresh": post.get("box_thresh", 0.6),
        "unclip_ratio": post.get("unclip_ratio", 1.5),
        "max_candidates": post.get("max_candidates", 1000),
    }


def load_rec_config(config_path):
    """
    Read the text recognition input shape and the CTC character list from the model's inference.yml.
    The character list starts with the CTC blank token and ends with a space.
    """
    config = _read_yaml(config_path)
    image_shape = _transform_op(config, "RecResizeImg").get("image_shape", [3, 48, 320])
    characters = [str(character) for character in config["PostProcess"]["character_dict"]]
    if " " not in characters:
        characters.append(" ")
    return {"image_shape": list(image_shape), "character_list": ["blank"] + characters}


def DetResizeForTest(img, limit_side_len=736, limit_type="min", max_side_limit=4000):
    """
    Resize an image for text detection, keeping the aspect ratio. Both sides become multiples of 32.

    Returns the resized image and the [src_h, src_w, ratio_h, ratio_w] shape array.
    """
    h, w = img.shape[:2]
    if limit_type == "max":
        ratio = float(limit_side_len) / max(h, w) if max(h, w) > limit_side_len else 1.0
    else:
        ratio = float(limit_side_len) / min(h, w) if min(h, w) < limit_side_len else 1.0
    resize_h, resize_w = int(h * ratio), int(w * ratio)
    if max_side_limit is not None and max(resize_h, resize_w) > max_side_limit:
        ratio = float(max_side_limit) / max(resize_h, resize_w)
        resize_h, resize_w = int(resize_h * ratio), int(resize_w * ratio)
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)
    img = cv2.resize(img, (resize_w, resize_h))
    return img, np.array([h, w, resize_h / float(h), resize_w / float(w)])


def letterbox(img, size):
    """
    Resize an image into a fixed `size` x `size` canvas, keeping the aspect ratio (static input shape).
    The image is placed in the top-left corner.

    Returns the resized image and its (height, width) inside the canvas.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    resize_h, resize_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return cv2.resize(img, (resize_w, resize_h)), (resize_h, resize_w)


def NormalizeImage(img, mean, std, scale=1.0 / 255.0):
    """Normalize an HWC image: (img * scale - mean) / std."""
    return (img.astype(np.float32) * scale - mean) / std


def unclip(box, unclip_ratio):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return offset.Execute(distance)


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def box_score_fast(bitmap, _box):
    """
    box_score_fast: use bbox mean score as the mean score
    """
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def boxes_from_bitmap(pred, _bitmap, dest_width, dest_height, box_thresh=0.6, unclip_ratio=1.5, max_candidates=1000):
    """
    DBPostProcess: convert the binarized probability map into text boxes in source image coordinates.

    pred: probability map with shape (H, W)
    _bitmap: binarized map with shape (H, W), values {0, 1}
    """

    bitmap = _bitmap
    height, width = bitmap.shape

    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[1] if len(outs) == 3 else outs[0]

    num_contours = min(len(contours), max_candidates)

    boxes = []
    scores = []
    for index in range(num_contours):
        contour = contours[index]
        points, sside = get_mini_boxes(contour)
        if sside < 3:
            continue
        points = np.array(points)
        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue

        expanded = unclip(points, unclip_ratio)
        if len(expanded) != 1:
            continue
        box, sside = get_mini_boxes(np.array(expanded[0]).reshape(-1, 1, 2))
        if sside < 3 + 2:
            continue
        box = np.array(box)
        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.astype(np.int32))
        scores.append(score)
    return np.array(boxes, dtype=np.int32).reshape(-1, 4, 2), scores


def filter_tag_det_res(dt_boxes, image_shape):
    img_height, img_width = image_shape[0:2]
    dt_boxes_new = []
    for box in dt_boxes:
        box = order_points_clockwise(box)
        box = clip_det_res(box, img_height, img_width)
        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))
        if rect_width <= 3 or rect_height <= 3:
            continue
        dt_boxes_new.append(box)
    dt_boxes = np.array(dt_boxes_new)
    return dt_boxes


def order_points_clockwise(pts):
    """
    reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
    # sort the points based on their x-coordinates
    """
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    tl, bl = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    tr, br = rightMost

    rect = np.array([tl, tr, br, bl], dtype="float32")
    return rect


def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points


def draw_text_det_res(dt_boxes, img_file):
    src_im = img_file
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.ascontiguousarray(np.rot90(dst_img))
    return dst_img


class CTCLabelDecode:
    """Convert CTC recognition model output into text and confidence."""

    def __init__(self, character_list):
        # character_list[0] is the CTC blank token
        self.character = character_list

    def __call__(self, preds):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        result_list = []
        for text_index, text_prob in zip(preds_idx, preds_prob):
            # remove repeated tokens and the blank token
            selection = np.ones(len(text_index), dtype=bool)
            selection[1:] = text_index[1:] != text_index[:-1]
            selection &= text_index != 0
            char_list = [self.character[int(index)] for index in text_index[selection]]
            conf_list = text_prob[selection]
            text = "".join(char_list)
            result_list.append((text, float(np.mean(conf_list)) if len(conf_list) else 0.0))
        return result_list


def draw_ocr_box_txt(image, boxes, txts, scores=None, drop_score=0.5, font_path="./fonts/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new("RGB", (w, h), (255, 255, 255))

    np.random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        box = [tuple(int(value) for value in point) for point in box]
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0],
                box[0][1],
                box[1][0],
                box[1][1],
                box[2][0],
                box[2][1],
                box[3][0],
                box[3][1],
            ],
            outline=color,
        )
        box_height = math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
        box_width = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size)
            cur_y = box[0][1]
            for c in txt:
                if hasattr(font, "getbbox"):
                    char_size = font.getbbox(c)
                    y_idx = -1
                else:
                    try:
                        char_size = font.getbox(c)
                        y_idx = -1
                    except BaseException:
                        char_size = font.getsize(c)
                        y_idx = 1

                draw_right.text((box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[y_idx]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size)
            draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)
