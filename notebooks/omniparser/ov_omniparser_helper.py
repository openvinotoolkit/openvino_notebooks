import json
from pathlib import Path
from typing import List, Optional, Union, Tuple
import types

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
import torch
import numpy as np

from ov_florence2_helper import download_original_model
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.ops import box_convert
from torchvision.transforms import ToPILImage
import supervision as sv
import openvino as ov

core = ov.Core()


def load_ov_icon_detector(model_path, device):
    from ultralytics import YOLO

    det_model = YOLO(model_path.parent, task="detect")

    ov_config = {}
    if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    compiled_model = core.compile_model(model_path, device, ov_config)

    if det_model.predictor is None:
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
        args = {**det_model.overrides, **custom}
        det_model.predictor = det_model._smart_load("predictor")(overrides=args, _callbacks=det_model.callbacks)
        det_model.predictor.setup_model(model=det_model.model)

    det_model.predictor.model.ov_compiled_model = compiled_model

    return det_model


def download_omniparser_florence_model():
    florence_caption_dir = Path("weights/icon_caption_florence")

    if not florence_caption_dir.exists():
        download_original_model("microsoft/Florence-2-base-ft", florence_caption_dir)
    pt_model = florence_caption_dir / "pytorch_model.bin"

    if pt_model.exists():
        pt_model.unlink()
        (pt_model.parent / "config.json").unlink()
    hf_hub_download("microsoft/OmniParser", filename="icon_caption_florence/model.safetensors", local_dir="weights")
    hf_hub_download("microsoft/OmniParser", filename="icon_caption_florence/config.json", local_dir="weights")
    hf_hub_download("microsoft/OmniParser", filename="icon_caption_florence/generation_config.json", local_dir="weights")

    with (florence_caption_dir / "config.json").open("r") as f:
        config_data = json.load(f)
    class_mapping = config_data["auto_map"]
    for key, value in class_mapping.items():
        class_mapping[key] = value.replace("microsoft/Florence-2-base-ft--", "")
    config_data["auto_map"] = class_mapping

    with (florence_caption_dir / "config.json").open("w") as f:
        json.dump(config_data, f)
    return florence_caption_dir


def download_omniparser_icon_detector():
    from notebook_utils import download_file

    icon_detector_dir = Path("weights/icon_detect")

    download_file("https://huggingface.co/spaces/microsoft/OmniParser/resolve/main/weights/icon_detect/best.pt", directory="weights/icon_detect")
    download_file("https://huggingface.co/spaces/microsoft/OmniParser/raw/main/weights/icon_detect/model.yaml", directory="weights/icon_detect")

    return icon_detector_dir


class BoxAnnotator:
    """
    A class for drawing bounding boxes on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to draw the bounding box,
            can be a single color or a color palette
        thickness (int): The thickness of the bounding box lines, default is 2
        text_color (Color): The color of the text on the bounding box, default is white
        text_scale (float): The scale of the text on the bounding box, default is 0.5
        text_thickness (int): The thickness of the text on the bounding box,
            default is 1
        text_padding (int): The padding around the text on the bounding box,
            default is 5

    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 3,  # 1 for seeclick 2 for mind2web and 3 for demo
        text_color: Color = Color.BLACK,
        text_scale: float = 0.5,  # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
        text_thickness: int = 2,  # 1, # 2 for demo
        text_padding: int = 10,
        avoid_overlap: bool = True,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.avoid_overlap: bool = avoid_overlap

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Draws bounding boxes on the frame using the detections provided.

        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the
                bounding boxes will be drawn
            labels (Optional[List[str]]): An optional list of labels
                corresponding to each detection. If `labels` are not provided,
                corresponding `class_id` will be used as label.
            skip_label (bool): Is set to `True`, skips bounding box label annotation.
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it

        Example:
            ```python
            import supervision as sv

            classes = ['person', ...]
            image = ...
            detections = sv.Detections(...)

            box_annotator = sv.BoxAnnotator()
            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _ in detections
            ]
            annotated_frame = box_annotator.annotate(
                scene=image.copy(),
                detections=detections,
                labels=labels
            )
            ```
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = detections.class_id[i] if detections.class_id is not None else None
            idx = class_id if class_id is not None else i
            color = self.color.by_idx(idx) if isinstance(self.color, ColorPalette) else self.color
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
            if skip_label:
                continue

            text = f"{class_id}" if (labels is None or len(detections) != len(labels)) else labels[i]

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            if not self.avoid_overlap:
                text_x = x1 + self.text_padding
                text_y = y1 - self.text_padding

                text_background_x1 = x1
                text_background_y1 = y1 - 2 * self.text_padding - text_height

                text_background_x2 = x1 + 2 * self.text_padding + text_width
                text_background_y2 = y1
                # text_x = x1 - self.text_padding - text_width
                # text_y = y1 + self.text_padding + text_height
                # text_background_x1 = x1 - 2 * self.text_padding - text_width
                # text_background_y1 = y1
                # text_background_x2 = x1
                # text_background_y2 = y1 + 2 * self.text_padding + text_height
            else:
                text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2 = get_optimal_label_pos(
                    self.text_padding, text_width, text_height, x1, y1, x2, y2, detections, image_size
                )

            cv2.rectangle(
                img=scene,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            # import pdb; pdb.set_trace()
            box_color = color.as_rgb()
            luminance = 0.299 * box_color[0] + 0.587 * box_color[1] + 0.114 * box_color[2]
            text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                # color=self.text_color.as_rgb(),
                color=text_color,
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def IoU(box1, box2, return_max=True):
    intersection = intersection_area(box1, box2)
    union = box_area(box1) + box_area(box2) - intersection
    if box_area(box1) > 0 and box_area(box2) > 0:
        ratio1 = intersection / box_area(box1)
        ratio2 = intersection / box_area(box2)
    else:
        ratio1, ratio2 = 0, 0
    if return_max:
        return max(intersection / union, ratio1, ratio2)
    else:
        return intersection / union


def get_optimal_label_pos(text_padding, text_width, text_height, x1, y1, x2, y2, detections, image_size):
    """check overlap of text and background detection box, and get_optimal_label_pos,
    pos: str, position of the text, must be one of 'top left', 'top right', 'outer left', 'outer right' TODO: if all are overlapping, return the last one, i.e. outer right
    Threshold: default to 0.3
    """

    def get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2, text_background_y2, image_size):
        is_overlap = False
        for i in range(len(detections)):
            detection = detections.xyxy[i].astype(int)
            if IoU([text_background_x1, text_background_y1, text_background_x2, text_background_y2], detection) > 0.3:
                is_overlap = True
                break
        # check if the text is out of the image
        if text_background_x1 < 0 or text_background_x2 > image_size[0] or text_background_y1 < 0 or text_background_y2 > image_size[1]:
            is_overlap = True
        return is_overlap

    # if pos == 'top left':
    text_x = x1 + text_padding
    text_y = y1 - text_padding

    text_background_x1 = x1
    text_background_y1 = y1 - 2 * text_padding - text_height

    text_background_x2 = x1 + 2 * text_padding + text_width
    text_background_y2 = y1
    is_overlap = get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2, text_background_y2, image_size)
    if not is_overlap:
        return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2

    # elif pos == 'outer left':
    text_x = x1 - text_padding - text_width
    text_y = y1 + text_padding + text_height

    text_background_x1 = x1 - 2 * text_padding - text_width
    text_background_y1 = y1

    text_background_x2 = x1
    text_background_y2 = y1 + 2 * text_padding + text_height
    is_overlap = get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2, text_background_y2, image_size)
    if not is_overlap:
        return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2

    # elif pos == 'outer right':
    text_x = x2 + text_padding
    text_y = y1 + text_padding + text_height

    text_background_x1 = x2
    text_background_y1 = y1

    text_background_x2 = x2 + 2 * text_padding + text_width
    text_background_y2 = y1 + 2 * text_padding + text_height

    is_overlap = get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2, text_background_y2, image_size)
    if not is_overlap:
        return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2

    # elif pos == 'top right':
    text_x = x2 - text_padding - text_width
    text_y = y1 - text_padding

    text_background_x1 = x2 - 2 * text_padding - text_width
    text_background_y1 = y1 - 2 * text_padding - text_height

    text_background_x2 = x2
    text_background_y2 = y1

    is_overlap = get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2, text_background_y2, image_size)
    if not is_overlap:
        return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2

    return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2


def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h


def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp


def get_xywh_yolo(input):
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h


def check_ocr_box(reader, image_path, output_bb_format="xywh", goal_filtering=None, easyocr_args=None):
    result = reader.readtext(image_path, **easyocr_args)
    # print('goal filtering pred:', result[-5:])
    coord = [item[0] for item in result]
    text = [item[1] for item in result]
    if output_bb_format == "xywh":
        bb = [get_xywh(item) for item in coord]
    elif output_bb_format == "xyxy":
        bb = [get_xyxy(item) for item in coord]
    return (text, bb), goal_filtering


def predict_yolo(model, image_path, box_threshold, imgsz):
    """Use huggingface model to replace the original model"""
    # model = model['model']

    result = model.predict(
        source=image_path,
        conf=box_threshold,
        imgsz=imgsz,
        # iou=0.5, # default 0.7
    )
    boxes = result[0].boxes.xyxy  # .tolist() # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases


def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(boxes):
        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            # add the following 2 lines to include ocr bbox
            if ocr_bbox:
                if not any(IoU(box1, box3) > iou_threshold for k, box3 in enumerate(ocr_bbox)):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
    return torch.tensor(filtered_boxes)


@torch.inference_mode()
def get_parsed_content_icon(filtered_boxes, ocr_bbox, image_source, caption_model_processor, prompt=None):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox) :]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0] * image_source.shape[1]), int(coord[2] * image_source.shape[1])
        ymin, ymax = int(coord[1] * image_source.shape[0]), int(coord[3] * image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor["model"], caption_model_processor["processor"]
    if not prompt:
        prompt = "<CAPTION>"

    batch_size = 10  # Number of samples per batch
    generated_texts = []

    for i in range(0, len(croped_pil_image), batch_size):
        batch = croped_pil_image[i : i + batch_size]
        inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt")
        if "florence" in model.config.name_or_path:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3, do_sample=False
            )
        else:
            generated_ids = model.generate(
                **inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, num_return_sequences=1
            )  # temperature=0.01, do_sample=True,
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)

    return generated_texts


def annotate(
    image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, text_padding=5, text_thickness=2, thickness=3
) -> np.ndarray:
    """
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]
    box_annotator = BoxAnnotator(
        text_scale=text_scale, text_padding=text_padding, text_thickness=text_thickness, thickness=thickness
    )  # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w, h))

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates


def get_som_labeled_img(
    img_path,
    model=None,
    BOX_TRESHOLD=0.01,
    output_coord_in_ratio=False,
    ocr_bbox=None,
    text_scale=0.4,
    text_padding=5,
    draw_bbox_config=None,
    caption_model_processor=None,
    ocr_text=[],
    use_local_semantics=True,
    iou_threshold=0.9,
    prompt=None,
    imgsz=640,
):
    """ocr_bbox: list of xyxy format bbox"""
    # TEXT_PROMPT = "clickable buttons on the screen"
    # # BOX_TRESHOLD = 0.02 # 0.05/0.02 for web and 0.1 for mobile
    # TEXT_TRESHOLD = 0.01 # 0.9 # 0.01
    image_source = Image.open(img_path).convert("RGB")
    w, h = image_source.size
    # import pdb; pdb.set_trace()
    xyxy, logits, phrases = predict_yolo(model=model, image_path=img_path, box_threshold=BOX_TRESHOLD, imgsz=imgsz)
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    image_source = np.asarray(image_source)
    phrases = [str(i) for i in range(len(phrases))]

    # annotate the image with labels
    h, w, _ = image_source.shape
    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox = ocr_bbox.tolist()
    else:
        print("no ocr bbox!!!")
        ocr_bbox = None
    filtered_boxes = remove_overlap(boxes=xyxy, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox)

    # get parsed icon local semantics
    if use_local_semantics:
        parsed_content_icon = get_parsed_content_icon(filtered_boxes, ocr_bbox, image_source, caption_model_processor, prompt=prompt)
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text)
        parsed_content_icon_ls = []
        for i, txt in enumerate(parsed_content_icon):
            parsed_content_icon_ls.append(f"Icon Box ID {str(i+icon_start)}: {txt}")
        parsed_content_merged = ocr_text + parsed_content_icon_ls
    else:
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        parsed_content_merged = ocr_text

    filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")

    phrases = [i for i in range(len(filtered_boxes))]

    # draw boxes
    if draw_bbox_config:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, **draw_bbox_config)
    else:
        annotated_frame, label_coordinates = annotate(
            image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, text_scale=text_scale, text_padding=text_padding
        )

    pil_img = Image.fromarray(annotated_frame)
    if output_coord_in_ratio:
        # h, w, _ = image_source.shape
        label_coordinates = {k: [v[0] / w, v[1] / h, v[2] / w, v[3] / h] for k, v in label_coordinates.items()}
        assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

    return pil_img, label_coordinates, parsed_content_merged


def process_image(image_path, icon_detector, caption_model_processor, reader, box_threshold=0.05, iou_threshold=0.1, imgsz=640):
    image = Image.open(image_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, _ = check_ocr_box(
        reader, str(image_path), output_bb_format="xyxy", goal_filtering=None, easyocr_args={"paragraph": False, "text_threshold": 0.9}
    )
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, labels_coordinates, parsed_content_list = get_som_labeled_img(
        image_path,
        icon_detector,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )
    print("finish processing")
    parsed_content_list = "\n".join(parsed_content_list)
    return dino_labled_img, labels_coordinates, str(parsed_content_list)


def easyocr_reader(models_dir, detector_device, recognizer_device):
    import easyocr

    reader = easyocr.Reader(["en"], quantize=False, gpu=False)
    recognizer_path = Path(models_dir) / "recognizer.xml"
    detector_path = Path(models_dir) / "detector.xml"

    if not recognizer_path.exists():
        ov_model = ov.convert_model(reader.recognizer, example_input=(torch.zeros([1, 1, 64, 320]), torch.zeros([1, 33], dtype=torch.long)))
        ov.save_model(ov_model, recognizer_path)

    if not detector_path.exists():
        ov_model = ov.convert_model(reader.detector, example_input=torch.zeros([1, 3, 1728, 2560]))
        ov.save_model(ov_model, detector_path)

    ov_recognizer = core.compile_model(recognizer_path, recognizer_device)
    ov_detector = core.compile_model(detector_path, detector_device)

    def forward_detector(self, input):
        result = ov_detector(input)
        return torch.from_numpy(result[0]), torch.from_numpy(result[1])

    def forward_recognizer(self, input, text):
        result = ov_recognizer([input, text])[0]
        return torch.from_numpy(result)

    reader.detector.forward = types.MethodType(forward_detector, reader.detector)
    reader.recognizer.forward = types.MethodType(forward_recognizer, reader.recognizer)

    return reader
