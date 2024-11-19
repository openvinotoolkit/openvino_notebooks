from pathlib import Path
import shutil
import torch
from PIL import Image


def converted_model_exists(model_dir):
    for file_name in ["openvino_language_model.xml", "openvino_text_embeddings_model.xml", "openvino_vision_embeddings_model.xml"]:
        if not (Path(model_dir) / file_name).exists() or not (Path(model_dir) / file_name.replace(".bin")).exists():
            return False

    return True


def copy_model_files(src_dir, dst_dir, ignore_llm=True, ignore_vision_encoder=True):
    ignore_files = []
    if ignore_llm:
        ignore_files.extend(["openvino_language_model.xml", "openvino_language_model.bin"])
    if ignore_vision_encoder:
        ignore_files.extend(["openvino_vision_embeddings_model.xml", "openvino_vision_embeddings_model.bin"])

    for file_name in src_dir.glob("*"):
        if file_name.name in ignore_files:
            continue
        shutil.copy(file_name, dst_dir)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, model_cfg, processor):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    else:
        return processor(images=images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def process_text_input(text, tokenizer):
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split("<image>")]
    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
    return input_ids, attention_mask
