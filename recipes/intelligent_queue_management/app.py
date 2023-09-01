import argparse
import json
import logging as log
import time
from collections import defaultdict, deque
from typing import Tuple, List

import cv2
import numpy as np
import supervision as sv
import torch
from openvino import runtime as ov
from ultralytics.yolo.utils import ops

import utils


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
        img = cv2.resize(img, dsize=new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))  # add border
    return img, ratio, (dw, dh)


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
    image = letterbox(image, new_shape=input_size)[0]
    # convert to float32
    image = image.astype(np.float32)
    # normalize to (0, 1)
    image /= 255.0
    # changes data layout from HWC to CHW
    image = image.transpose((2, 0, 1))
    # add one more dimension
    image = np.expand_dims(image, axis=0)
    return image


def postprocess(pred_boxes: np.ndarray, input_size: Tuple[int, int], orig_img, min_conf_threshold=0.25, nms_iou_threshold=0.75, agnostic_nms=False, max_detections=100) -> sv.Detections:
    """
        YOLOv8 model postprocessing function. Applied non-maximum supression algorithm to detections and rescale boxes to original image size,
         filtering out other classes than person

        Parameters:
            pred_boxes: model output prediction boxes
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
    # non-maximum suppresion
    pred = ops.non_max_suppression(torch.from_numpy(pred_boxes), min_conf_threshold, nms_iou_threshold, nc=80, **nms_kwargs)[0]

    # no predictions in the image
    if not len(pred):
        return sv.Detections.empty()

    # transform boxes to pixel coordinates
    pred[:, :4] = ops.scale_boxes(input_size, pred[:, :4], orig_img.shape).round()
    # numpy array from torch tensor
    pred = np.array(pred)
    # create detections in supervision format
    det = sv.Detections(pred[:, :4], confidence=pred[:, 4], class_id=pred[:, 5])
    # filter out other predictions than people
    return det[det.class_id == 0]


def get_model(model_path: str) -> ov.CompiledModel:
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
    model = core.compile_model(model, device_name="AUTO", config={"PERFORMANCE_HINT": "LATENCY"})

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
    for index, polygon in enumerate(polygons):
        # a zone to count people in
        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=resolution_wh)
        zones.append(zone)
        # the annotator - visual part of the zone
        zone_annotators.append(sv.PolygonZoneAnnotator(zone=zone, color=colors.by_idx(index), thickness=4))
        # box annotator, showing boxes around people
        box_annotators.append(sv.BoxAnnotator(color=colors.by_idx(index)))

    return zones, zone_annotators, box_annotators


def draw_text(image: np.ndarray, text: str, point: tuple, color: tuple = (255, 255, 255)) -> None:
    """
    Draws "Store assistant required" in the bottom-right corner

    Parameters:
        image: image to draw on
        text: text to draw
        point: top left corner of the text
        color: text color
    """
    _, f_width = image.shape[:2]
    text_size, _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=f_width / 1500, thickness=2)

    rect_width = text_size[0] + 20
    rect_height = text_size[1] + 20
    rect_x, rect_y = point

    cv2.rectangle(image, pt1=(rect_x, rect_y), pt2=(rect_x + rect_width, rect_y + rect_height), color=(0, 0, 0), thickness=cv2.FILLED)

    text_x = rect_x + (rect_width - text_size[0]) // 2
    text_y = rect_y + (rect_height + text_size[1]) // 2

    cv2.putText(image, text=text, org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=f_width / 1500, color=color, thickness=2, lineType=cv2.LINE_AA)


def run(video_path: str, model_path: str, zones_config_file: str, customers_limit: int) -> None:
    """
    Runs main app

    Parameters:
        video_path: video path or camera number
        model_path: path to the exported model
        zones_config_file: path to config file for zones
        customers_limit: limit of customers in the queue
    """
    # set up logging
    log.getLogger().setLevel(log.INFO)

    # initialize and load model
    model = get_model(model_path)
    # input shape of the model (w, h, d)
    input_shape = tuple(model.inputs[0].shape)[:0:-1]

    if video_path.isnumeric():
        video_path = int(video_path)

    # initialize video player to deliver frames
    player = utils.VideoPlayer(video_path, fps=60)

    # get zones, and zone and box annotators for zones
    zones, zone_annotators, box_annotators = get_annotators(json_path=zones_config_file, resolution_wh=(player.width, player.height))

    # people counter
    queue_count = defaultdict(deque)
    # keep at most 100 last times
    processing_times = deque(maxlen=100)

    # start a video stream
    player.start()
    while True:
        # get a frame
        frame = player.next()
        # if no more frames
        if frame is None:
            log.warning("The stream has ended")
            break

        f_height, f_width = frame.shape[:2]

        # preprocessing
        input_image = preprocess(image=frame, input_size=input_shape[:2])
        # prediction
        start_time = time.time()
        prediction = model(input_image)[model.outputs[0]]
        processing_times.append(time.time() - start_time)
        # postprocessing
        detections = postprocess(pred_boxes=prediction, input_size=input_shape[:2], orig_img=frame)

        # annotate the frame with the detected persons within each zone
        for zone_id, (zone, zone_annotator, box_annotator) in enumerate(zip(zones, zone_annotators, box_annotators)):
            # visualize polygon for the zone
            frame = zone_annotator.annotate(scene=frame)

            # get detections relevant only for the zone
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            # visualize boxes around people in the zone
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
            # count how many people detected
            det_count = len(detections_filtered)

            # add the count to the list
            queue_count[zone_id].append(det_count)
            # store the results from last 300 frames (approx. 5-10s)
            if len(queue_count[zone_id]) > 300:
                queue_count[zone_id].popleft()
            # calculate teh mean number of customers in the queue
            mean_customer_count = np.mean(queue_count[zone_id], dtype=np.int32)

            # add alert text to the frame if necessary
            if mean_customer_count > customers_limit:
                draw_text(frame, text=f"Store assistant required on cash desk {zone_id}!", point=(20, 20), color=(0, 0, 255))

            # print an info about number of customers in the queue, ask for the more assistants if required
            log.info(f"Checkout queue: {zone_id}, avg customer count: {mean_customer_count} {'Store assistant required!' if mean_customer_count > customers_limit else ''}")

        # Mean processing time [ms].
        processing_time = np.mean(processing_times) * 1000
        fps = 1000 / processing_time

        draw_text(frame, text=f"Inference time: {processing_time:.0f}ms ({fps:.1f} FPS)", point=(f_width * 3 // 5, 10))

        # show the output live
        cv2.imshow("Intelligent Queue Management System", frame)
        key = cv2.waitKey(1)
        # escape = 27 or 'q' to close the app
        if key == 27 or key == ord('q'):
            break

    # stop the stream
    player.stop()
    # clean-up windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', type=str, required=True, help="Path to a video file or the webcam number")
    parser.add_argument('--model_path', type=str, default="model/yolov8m_openvino_int8_model/yolov8m.xml", help="Path to the model")
    parser.add_argument('--zones_config_file', type=str, default="zones.json", help="Path to the zone config file (json)")
    parser.add_argument('--customers_limit', type=int, default=3, help="The maximum number of customers in the queue")

    args = parser.parse_args()
    run(args.stream, args.model_path, args.zones_config_file, args.customers_limit)
