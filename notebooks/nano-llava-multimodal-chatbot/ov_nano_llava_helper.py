from pathlib import Path
import shutil
from huggingface_hub import snapshot_download
import torch
from PIL import Image


def download_original_model(model_id, model_local_dir):
    if not model_local_dir.exists():
        snapshot_download(repo_id=model_id, local_dir=model_local_dir)

    modeling_file = model_local_dir / "modeling_llava_qwen2.py"
    orig_modeling_file = model_local_dir / f"orig_{modeling_file.name}"

    # model code depends from flash_attn package that may be problematic to load. Patch model code for avoiding import of this package
    if not orig_modeling_file.exists():
        modeling_file.rename(orig_modeling_file)
    with orig_modeling_file.open("r") as f:
        content = f.read()
    replacement_lines = [
        ("from flash_attn import flash_attn_func, flash_attn_varlen_func", ""),
        ("from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input", ""),
        (' _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)', "pass"),
    ]

    for replace_pair in replacement_lines:
        content = content.replace(*replace_pair)

    with modeling_file.open("w") as f:
        f.write(content)


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
