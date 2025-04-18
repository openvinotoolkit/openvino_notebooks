from typing import Optional, Dict
import numpy as np
import cv2
from openvino.runtime.utils.data_helpers import OVDict
import matplotlib.pyplot as plt


def add_detection_box(box: np.ndarray, image: np.ndarray, label: Optional[str] = None, mask: np.ndarray = None) -> np.ndarray:
    """
    Helper function for adding single bounding box to the image

    Parameters
    ----------
    box : np.ndarray
        Bounding box coordinates in format [ymin, xmin, ymax, xmax]
    image : np.ndarray
        The image to which detection box is added
    label : str, optional
        Detection box label string, if not provided will not be added to result image (default is None)
    mask: np.ndarray
        Segmentation mask in format (H, W)

    Returns
    -------
    np.ndarray
        NumPy array including both image and detection box

    """
    ymin, xmin, ymax, xmax = box
    point1, point2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
    box_color = [np.random.randint(0, 255) for _ in range(3)]
    line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1

    result = cv2.rectangle(
        img=image,
        pt1=point1,
        pt2=point2,
        color=box_color,
        thickness=line_thickness,
        lineType=cv2.LINE_AA,
    )

    if label:
        font_thickness = max(line_thickness - 1, 1)
        font_face = 0
        font_scale = line_thickness / 3
        font_color = (255, 255, 255)
        text_size = cv2.getTextSize(
            text=label,
            fontFace=font_face,
            fontScale=font_scale,
            thickness=font_thickness,
        )[0]
        # Calculate rectangle coordinates
        rectangle_point1 = point1
        rectangle_point2 = (point1[0] + text_size[0], point1[1] - text_size[1] - 3)
        # Add filled rectangle
        result = cv2.rectangle(
            img=result,
            pt1=rectangle_point1,
            pt2=rectangle_point2,
            color=box_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        # Calculate text position
        text_position = point1[0], point1[1] - 3
        # Add text with label to filled rectangle
        result = cv2.putText(
            img=result,
            text=label,
            org=text_position,
            fontFace=font_face,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )
    if mask is not None:
        mask_img = mask[:, :, np.newaxis] * box_color
        result = cv2.addWeighted(result, 1, mask_img.astype(np.uint8), 0.6, 0)

    return result


def get_mask_frame(box, frame, mask):
    """
    Transform a binary mask to fit within a specified bounding box in a frame using perspective transformation.

    Args:
        box (tuple): A bounding box represented as a tuple (y_min, x_min, y_max, x_max).
        frame (numpy.ndarray): The larger frame or image where the mask will be placed.
        mask (numpy.ndarray): A binary mask image to be transformed.

    Returns:
        numpy.ndarray: A transformed mask image that fits within the specified bounding box in the frame.
    """
    x_min = frame.shape[1] * box[1]
    y_min = frame.shape[0] * box[0]
    x_max = frame.shape[1] * box[3]
    y_max = frame.shape[0] * box[2]
    rect_src = np.array(
        [
            [0, 0],
            [mask.shape[1], 0],
            [mask.shape[1], mask.shape[0]],
            [0, mask.shape[0]],
        ],
        dtype=np.float32,
    )
    rect_dst = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect_src[:, :], rect_dst[:, :])
    mask_frame = cv2.warpPerspective(mask, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_CUBIC)
    return mask_frame


def visualize_inference_result(
    inference_result: OVDict,
    image: np.ndarray,
    labels_map: Dict,
    detections_limit: Optional[int] = None,
):
    """
    Helper function for visualizing inference result on the image

    Parameters
    ----------
    inference_result : OVDict
        Result of the compiled model inference on the test image
    image : np.ndarray
        Original image to use for visualization
    labels_map : Dict
        Dictionary with mappings of detection classes numbers and its names
    detections_limit : int, optional
        Number of detections to show on the image, if not provided all detections will be shown (default is None)
    """
    detection_boxes = inference_result.get("detection_boxes")
    detection_classes = inference_result.get("detection_classes")
    detection_scores = inference_result.get("detection_scores")
    num_detections = inference_result.get("num_detections")
    detection_masks = inference_result.get("detection_masks")

    detections_limit = int(min(detections_limit, num_detections[0]) if detections_limit is not None else num_detections[0])

    # Normalize detection boxes coordinates to original image size
    original_image_height, original_image_width, _ = image.shape
    if detection_masks is not None:
        normalized_detection_boxes = detection_boxes[0, :detections_limit]
    else:
        normalized_detection_boxes = detection_boxes[::]
    normalized_detection_boxes = normalized_detection_boxes * [
        original_image_height,
        original_image_width,
        original_image_height,
        original_image_width,
    ]
    image_with_detection_boxes = np.copy(image)
    for i in range(detections_limit):
        detected_class_name = labels_map[int(detection_classes[0, i])]
        score = detection_scores[0, i]
        label = f"{detected_class_name} {score:.2f}"
        if detection_masks is not None:
            mask = detection_masks[0, i]
            mask_reframed = get_mask_frame(detection_boxes[0, i], image, mask)
            mask_reframed = (mask_reframed > 0.5).astype(np.uint8)
            image_with_detection_boxes = add_detection_box(box=normalized_detection_boxes[i], image=image_with_detection_boxes, label=label, mask=mask_reframed)
        else:
            image_with_detection_boxes = add_detection_box(
                box=normalized_detection_boxes[0, i],
                image=image_with_detection_boxes,
                label=label,
            )

    plt.imshow(image_with_detection_boxes)
