import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import openvino as ov
import torch
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ROOT = REPO_ROOT
GROUNDING_DINO_DIR = Path(os.environ.get("GROUNDING_DINO_DIR", REPO_ROOT / "GroundingDINO"))
sys.path.insert(0, str(GROUNDING_DINO_DIR))

from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
from groundingdino.util import get_tokenlizer
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap


def transform_image(pil_image: Image.Image) -> torch.Tensor:
    import groundingdino.datasets.transforms as transforms

    transform = transforms.Compose(
        [
            transforms.RandomResize([800], max_size=1333),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(pil_image.convert("RGB"), None)
    return image


def normalize_caption(caption: str) -> str:
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    return caption


def make_inputs(
    pil_image: Image.Image,
    caption: str,
    tokenizer,
    max_text_len: int,
    ground_dino_img_size: tuple[int, int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    caption = normalize_caption(caption)
    tokenized = tokenizer([caption], padding="longest", return_tensors="pt")
    special_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

    text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens, tokenizer
    )

    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
        position_ids = position_ids[:, :max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, :max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, :max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :max_text_len]

    from torchvision.transforms.functional import InterpolationMode, resize

    input_img = resize(
        transform_image(pil_image),
        list(ground_dino_img_size),
        interpolation=InterpolationMode.BICUBIC,
    )[None, ...]

    inputs = {
        "samples": input_img.numpy().astype(np.float32, copy=False),
        "input_ids": tokenized["input_ids"].numpy(),
        "attention_mask.1": tokenized["attention_mask"].numpy(),
        "position_ids": position_ids.numpy(),
        "token_type_ids": tokenized["token_type_ids"].numpy(),
        "text_self_attention_masks": text_self_attention_masks.numpy(),
    }
    phrase_tokens = tokenizer(caption)
    return inputs, phrase_tokens


def cxcywh_to_xyxy_pixels(boxes: torch.Tensor, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    boxes = boxes.clone()
    boxes = boxes * torch.tensor([width, height, width, height], dtype=boxes.dtype)
    boxes_xyxy = torch.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, width - 1)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, height - 1)
    return boxes_xyxy.numpy()


def draw_detections(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    line_width = max(2, min(8, annotated.width // 250))

    for box, label in zip(boxes_xyxy, labels):
        x0, y0, x1, y1 = [float(value) for value in box]
        draw.rectangle((x0, y0, x1, y1), outline=(255, 70, 40), width=line_width)
        label_bbox = draw.textbbox((x0, y0), label, font=font)
        label_w = label_bbox[2] - label_bbox[0]
        label_h = label_bbox[3] - label_bbox[1]
        label_y0 = max(0, y0 - label_h - 6)
        draw.rectangle((x0, label_y0, x0 + label_w + 8, label_y0 + label_h + 6), fill=(255, 70, 40))
        draw.text((x0 + 4, label_y0 + 3), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    positive = values >= 0
    negative = ~positive
    result = np.empty_like(values, dtype=np.float32)
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[negative])
    result[negative] = exp_values / (1 + exp_values)
    return result


def get_compiled_property(compiled_model: ov.CompiledModel, name: str):
    try:
        value = compiled_model.get_property(name)
    except Exception:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return str(value)


def get_core_property(core: ov.Core, device: str, name: str):
    try:
        value = core.get_property(device, name)
    except Exception:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return str(value)


def run(args: argparse.Namespace) -> dict:
    config = SLConfig.fromfile(str(args.config))
    tokenizer = get_tokenlizer.get_tokenlizer(config.text_encoder_type)
    max_text_len = config.max_text_len

    core = ov.Core()
    available_devices = [str(device) for device in core.available_devices]
    full_device_name = get_core_property(core, args.device, "FULL_DEVICE_NAME")
    model = core.read_model(args.model)
    compile_start = perf_counter()
    compiled_model = core.compile_model(model, args.device)
    compile_ms = (perf_counter() - compile_start) * 1000
    execution_devices = get_compiled_property(compiled_model, "EXECUTION_DEVICES")
    inference_precision_hint = get_compiled_property(compiled_model, "INFERENCE_PRECISION_HINT")

    image = Image.open(args.image).convert("RGB")

    preprocess_start = perf_counter()
    inputs, phrase_tokens = make_inputs(image, args.prompt, tokenizer, max_text_len, tuple(args.ground_dino_img_size))
    preprocess_ms = (perf_counter() - preprocess_start) * 1000

    infer_request = compiled_model.create_infer_request()
    for _ in range(args.warmup):
        infer_request.infer(inputs, share_inputs=False)

    inference_times_ms = []
    inference_start = perf_counter()
    for _ in range(args.repeat):
        iteration_start = perf_counter()
        infer_request.start_async(inputs, share_inputs=False)
        infer_request.wait()
        inference_times_ms.append((perf_counter() - iteration_start) * 1000)
    inference_ms = (perf_counter() - inference_start) * 1000 / args.repeat
    latency_until_postprocess_ms = preprocess_ms + inference_ms

    postprocess_start = perf_counter()
    logits = torch.from_numpy(sigmoid(np.squeeze(infer_request.get_tensor("pred_logits").data, 0)))
    boxes = torch.from_numpy(np.squeeze(infer_request.get_tensor("pred_boxes").data, 0))

    scores = logits.max(dim=1)[0]
    keep = scores > args.box_threshold
    logits = logits[keep]
    boxes = boxes[keep]
    scores = scores[keep]

    labels = []
    for logit, score in zip(logits, scores):
        phrase = get_phrases_from_posmap(logit > args.text_threshold, phrase_tokens, tokenizer)
        labels.append(f"{phrase} {score.item():.2f}".strip())

    boxes_xyxy = cxcywh_to_xyxy_pixels(boxes, image.size) if len(boxes) else np.empty((0, 4), dtype=np.float32)
    draw_detections(image, boxes_xyxy, labels, args.output)
    postprocess_ms = (perf_counter() - postprocess_start) * 1000

    result = {
        "image": str(args.image),
        "prompt": args.prompt,
        "device": args.device,
        "requested_device": args.device,
        "openvino_available_devices": available_devices,
        "openvino_full_device_name": full_device_name,
        "openvino_execution_devices": execution_devices,
        "openvino_inference_precision_hint": inference_precision_hint,
        "compile_ms": compile_ms,
        "preprocess_ms": preprocess_ms,
        "openvino_inference_ms": inference_ms,
        "openvino_inference_times_ms": inference_times_ms,
        "latency_until_postprocess_ms": latency_until_postprocess_ms,
        "latency_before_postprocess_ms": latency_until_postprocess_ms,
        "fps_until_postprocess": 1000 / latency_until_postprocess_ms if latency_until_postprocess_ms else None,
        "fps_before_postprocess": 1000 / latency_until_postprocess_ms if latency_until_postprocess_ms else None,
        "openvino_inference_fps": 1000 / inference_ms if inference_ms else None,
        "postprocess_ms": postprocess_ms,
        "request_total_ms_excluding_compile": preprocess_ms + inference_ms + postprocess_ms,
        "num_detections": len(labels),
        "detections": [
            {"label": label, "box_xyxy": [round(float(value), 2) for value in box]}
            for label, box in zip(labels, boxes_xyxy)
        ],
        "output_image": str(args.output),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one GroundingDINO OpenVINO request and save an annotated image.")
    parser.add_argument("--model", type=Path, default=ROOT / "openvino_irs/openvino_grounding_dino.xml")
    parser.add_argument("--config", type=Path, default=GROUNDING_DINO_DIR / "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--image", type=Path, default=Path("/home/ubuntu/Test_Images/Got7_A.jpg"))
    parser.add_argument("--prompt", default="Person with white hair")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--ground-dino-img-size", type=int, nargs=2, default=(1024, 1280), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/gdino_got7_white_hair.jpg")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup iterations excluded from latency reporting.")
    parser.add_argument("--repeat", type=int, default=1, help="Measured inference iterations. Use 1 for a single request.")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))