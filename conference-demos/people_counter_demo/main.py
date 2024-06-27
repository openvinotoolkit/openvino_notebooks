import argparse
import json
import logging as log
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import supervision as sv
import torch
from openvino import runtime as ov
from ultralytics import YOLO
from ultralytics.utils import ops

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils


def convert(det_model_name: str, model_dir: Path) -> tuple[Path, Path]:
    """
    Convert YOLO model

    Params:
        det_model_name: name of the YOLO model we want to use
        model_dir: dir to export model
        quantize: whether to quantize
    Returns:
       Path to exported model
    """
    model_path = model_dir / f"{det_model_name}.pt"
    # create a YOLO object detection model
    det_model = YOLO(model_path)

    ov_model_path = model_dir / f"{det_model_name}_openvino_model"
    ov_int8_model_path = model_dir / f"{det_model_name}_int8_openvino_model"
    # export the model to OpenVINO format (FP16 and INT8)
    if not ov_model_path.exists():
        ov_model_path = det_model.export(format="openvino", dynamic=False, half=True)
    if not ov_int8_model_path.exists():
        ov_int8_model_path = det_model.export(format="openvino", dynamic=False, half=True, int8=True)
    return ov_model_path / f"{det_model_name}.xml", ov_int8_model_path / f"{det_model_name}.xml"


def letterbox(img: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
        Resize image and padding for detection. Takes image as input,
         resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints

        Parameters:
          img: image for preprocessing
          new_shape: image size after preprocessing in format [width, height]
        Returns:
          img: image after preprocessing
          ratio: hight and width scaling ratio
          padding_size: height and width padding size
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[1::-1]  # current shape [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape != new_unpad:  # resize
        img = cv2.resize(img, dsize=new_unpad, interpolation=cv2.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))  # add border
    return img, ratio, (int(dw), int(dh))


def preprocess(image: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    """
        Preprocess image according to YOLOv8 input requirements.

        Parameters:
          image: image for preprocessing
          input_size: image size after preprocessing in format [width, height]
        Returns:
          img: image after preprocessing
    """
    # add padding to the image
    image, _, padding = letterbox(image, new_shape=input_size)
    # convert to float32
    image = image.astype(np.float32)
    # normalize to (0, 1)
    image /= 255.0
    # changes data layout from HWC to CHW
    image = image.transpose((2, 0, 1))
    # add one more dimension
    image = np.expand_dims(image, axis=0)
    return image, padding


def postprocess(pred_boxes: np.ndarray, pred_masks: np.ndarray, input_size: Tuple[int, int], orig_img, padding, min_conf_threshold=0.25, nms_iou_threshold=0.75, agnostic_nms=False, max_detections=100) -> sv.Detections:
    """
        YOLOv8 model postprocessing function. Applied non-maximum supression algorithm to detections and rescale boxes to original image size,
         filtering out other classes than person

         Parameters:
            pred_boxes: model output prediction boxes
            pred_masks: model output prediction masks
            input_size: image size after preprocessing in format [width, height]
            orig_img: image before preprocessing
            min_conf_threshold: minimal accepted confidence for object filtering
            nms_iou_threshold: minimal overlap score for removing objects duplicates in NMS
            agnostic_nms: apply class agnostinc NMS approach or not
            max_detections:  maximum detections after NMS
         Returns:
            det: list of detected boxes in sv.Detections format
    """
    nms_kwargs = {"agnostic": agnostic_nms, "max_det": max_detections}
    # non-maximum suppression
    pred = ops.non_max_suppression(torch.from_numpy(pred_boxes), min_conf_threshold, nms_iou_threshold, nc=80, **nms_kwargs)[0]

    # no predictions in the image
    if not len(pred):
        return sv.Detections.empty()

    masks = pred_masks
    if pred_masks is not None:
        # upscale masks
        masks = np.array(ops.process_mask(torch.from_numpy(pred_masks[0]), pred[:, 6:], pred[:, :4], input_size, upsample=True))
        masks = np.array([cv2.resize(mask[padding[1]:-padding[1] - 1, padding[0]:-padding[0] - 1], orig_img.shape[:2][::-1], interpolation=cv2.INTER_AREA) for mask in masks])
    # transform boxes to pixel coordinates
    pred[:, :4] = ops.scale_boxes(input_size, pred[:, :4], orig_img.shape).round()
    # numpy array from torch tensor
    pred = np.array(pred)
    # create detections in supervision format
    det = sv.Detections(xyxy=pred[:, :4], mask=masks, confidence=pred[:, 4], class_id=pred[:, 5])
    # filter out other predictions than people
    return det[det.class_id == 0]


def get_model(model_path: str, device: str = "AUTO") -> ov.CompiledModel:
    """
        Initialize OpenVINO and compile model for latency processing

        Parameters:
            model_path: path to the model to load
        Returns:
           model: compiled and ready OpenVINO model
    """
    # initialize OpenVINO
    core = ov.Core()
    # read the model from file
    model = core.read_model(model_path)
    # compile the model for latency mode
    model = core.compile_model(model, device_name=device, config={"PERFORMANCE_HINT": "LATENCY"})

    return model


def load_zones(json_path: str) -> List[np.ndarray]:
    """
        Load zones specified in an external json file

        Parameters:
            json_path: path to the json file with defined zones
        Returns:
           zones: a list of arrays with zone points
    """
    # load json file
    with open(json_path) as f:
        zones_dict = json.load(f)

    # return a list of zones defined by points
    return [np.array(zone["points"], np.int32) for zone in zones_dict.values()]


def get_annotators(json_path: str, resolution_wh: Tuple[int, int]) -> Tuple[List, List, List]:
    """
        Load zones specified in an external json file

        Parameters:
            json_path: path to the json file with defined zones
            resolution_wh: width and height of the frame
        Returns:
           zones, zone_annotators, box_annotators: lists of zones and their annotators
    """
    # list of points
    polygons = load_zones(json_path)

    # colors for zones
    colors = sv.ColorPalette.default()

    zones = []
    zone_annotators = []
    box_annotators = []
    masks_annotators = []
    for index, polygon in enumerate(polygons):
        # a zone to count people in
        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=resolution_wh)
        zones.append(zone)
        # the annotator - visual part of the zone
        zone_annotators.append(sv.PolygonZoneAnnotator(zone=zone, color=colors.by_idx(index), thickness=4))
        # box annotator, showing boxes around people
        box_annotators.append(sv.BoxAnnotator(color=colors.by_idx(index)))
        # mask annotator, showing transparent mask
        masks_annotators.append(sv.MaskAnnotator(color=colors.by_idx(index)))

    return zones, zone_annotators, box_annotators, masks_annotators


def draw_text(image, text, point, color=(255, 255, 255)) -> None:
    """
    Draws "Store assistant required" in the bottom-right corner

    Parameters:
        image: image to draw on
        text: text to draw
        point: top left corner of the text
        color: text color
    """
    _, f_width = image.shape[:2]
    text_size, _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=f_width / 2000, thickness=2)

    rect_width = text_size[0] + 50
    rect_height = text_size[1] + 30
    rect_x, rect_y = point

    cv2.rectangle(image, pt1=(rect_x, rect_y), pt2=(rect_x + rect_width, rect_y + rect_height), color=(0, 0, 0), thickness=cv2.FILLED)

    text_x = rect_x + (rect_width - text_size[0]) // 2
    text_y = rect_y + (rect_height + text_size[1]) // 2

    cv2.putText(image, text=text, org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=f_width / 2000, color=color, thickness=1, lineType=cv2.LINE_AA)


def draw_info(image, device_mapping):
    h, w = image.shape[:2]
    line_space = 40
    start_y = (len(device_mapping) + 3) * line_space + 20
    draw_text(image, "Control panel. Press:", (10, h - start_y))
    draw_text(image, "f: FP16 model", (10, h - start_y + line_space))
    draw_text(image, "i: INT8 model", (10, h - start_y + 2 * line_space))
    for i, (device_name, device_info) in enumerate(device_mapping.items(), start=1):
        draw_text(image, f"{i}: {device_name} - {device_info}", (10, h - start_y + (i + 2) * line_space))


def run(video_path: str, model_paths: Tuple[Path, Path], zones_config_file: str, people_limit: int = 3, last_frames: int = 50, model_name: str = "") -> None:
    """
    Runs main app

    Parameters:
        video_path: video path or camera number
        model_paths: paths to the exported models
        zones_config_file: path to config file for zones
        customers_limit: limit of customers in the queue
    """
    # set up logging
    log.getLogger().setLevel(log.INFO)

    MODEL_MAPPING = {
        "FP16": model_paths[0],
        "INT8": model_paths[1],
    }

    DEVICE_MAPPING = {"AUTO": "AUTO device"}

    core = ov.Core()
    for device in core.available_devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        if "nvidia" not in device_name.lower():
            DEVICE_MAPPING[device] = device_name

    model_type = "INT8"
    device_type = "AUTO"

    core.set_property({"CACHE_DIR": "cache"})
    # initialize and load model
    model = get_model(MODEL_MAPPING[model_type], device_type)
    # input shape of the model (w, h, d)
    input_shape = tuple(model.inputs[0].shape)[:0:-1]

    # initialize video player to deliver frames
    if isinstance(video_path, str) and video_path.isnumeric():
        video_path = int(video_path)
    player = utils.VideoPlayer(video_path, size=(1920, 1080), fps=60)

    # get zones, and zone and box annotators for zones
    zones, zone_annotators, box_annotators, masks_annotators = get_annotators(json_path=zones_config_file,
                                                                              resolution_wh=(
                                                                              player.width, player.height))

    # people counter
    queue_count = defaultdict(lambda: deque(maxlen=last_frames))
    # keep at most 100 last times
    processing_times = deque(maxlen=100)

    title = "Press ESC to Exit"
    cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # start a video stream
    player.start()
    while True:
        # Grab the frame.
        frame = player.next()
        if frame is None:
            print("Source ended")
            break
        # Get the results.
        frame = np.array(frame)
        f_height, f_width = frame.shape[:2]

        # preprocessing
        input_image, padding = preprocess(image=frame, input_size=input_shape[:2])
        # prediction
        start_time = time.time()
        results = model(input_image)
        processing_times.append(time.time() - start_time)
        boxes = results[model.outputs[0]]
        masks = results[model.outputs[1]] if len(model.outputs) > 1 else None
        # postprocessing
        detections = postprocess(pred_boxes=boxes, pred_masks=masks, input_size=input_shape[:2], orig_img=frame,
                                 padding=padding)

        # annotate the frame with the detected persons within each zone
        for zone_id, (zone, zone_annotator, box_annotator, masks_annotator) in enumerate(
                zip(zones, zone_annotators, box_annotators, masks_annotators), start=1):
            # visualize polygon for the zone
            frame = zone_annotator.annotate(scene=frame)

            # get detections relevant only for the zone
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            # visualize boxes around people in the zone - uncomment if you want to draw masks
            # frame = masks_annotator.annotate(scene=frame, detections=detections_filtered)
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
            # count how many people detected
            det_count = len(detections_filtered)

            # add the count to the list
            queue_count[zone_id].append(det_count)
            # calculate the mean number of customers in the queue
            mean_customer_count = np.mean(queue_count[zone_id], dtype=np.int32)

            # add alert text to the frame if necessary, flash every second
            if mean_customer_count > people_limit and time.time() % 2 > 1:
                draw_text(frame, text=f"Intel employee required in zone {zone_id}!", point=(20, 20), color=(0, 0, 255))

            # print an info about number of customers in the queue, ask for the more assistants if required
            log.info(
                f"Zone {zone_id}, avg people count: {mean_customer_count} {'Intel employee required!' if mean_customer_count > people_limit else ''}")

        # Mean processing time [ms].
        processing_time = np.mean(processing_times) * 1000

        fps = 1000 / processing_time
        draw_text(frame, text=f"Inference time: {processing_time:.0f}ms ({fps:.1f} FPS)", point=(f_width * 3 // 5, 10))
        draw_text(frame, text=f"Currently running {model_name} ({model_type}) on {device_type}",
                  point=(f_width * 3 // 5, 50))

        draw_info(frame, DEVICE_MAPPING)
        utils.draw_ov_watermark(frame)
        # show the output live
        cv2.imshow(title, frame)
        key = cv2.waitKey(1)
        # escape = 27 or 'q' to close the app
        if key == 27 or key == ord('q'):
            break

        model_changed = False
        if key == ord('f'):
            model_type = "FP16"
            model_changed = True
        if key == ord('i'):
            model_type = "INT8"
            model_changed = True
        for i, dev in enumerate(DEVICE_MAPPING.keys()):
            if key == ord('1') + i:
                device_type = dev
                model_changed = True

        if model_changed:
            del model
            model = get_model(MODEL_MAPPING[model_type], device_type)
            processing_times.clear()

    # stop the stream
    player.stop()
    # clean-up windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default="0", type=str, help="Path to a video file or the webcam number")
    parser.add_argument("--model_name", type=str, choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                        default="yolov8n", help="Model version to be converted")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
    parser.add_argument('--zones_config_file', type=str, default="zones.json", help="Path to the zone config file (json)")
    parser.add_argument('--people_limit', type=int, default=3, help="The maximum number of people in the area")

    args = parser.parse_args()
    model_paths = convert(args.model_name, Path(args.model_dir))
    run(args.stream, model_paths, args.zones_config_file, args.people_limit, model_name=args.model_name)
