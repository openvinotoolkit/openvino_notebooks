import torch
from datasets import load_dataset
from transformers import AutoProcessor
from tqdm.autonotebook import tqdm
from pathlib import Path
import pickle
import gc

import requests
from io import BytesIO
import numpy as np
from PIL import Image
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from ov_mllama_helper import OVMLlamaForConditionalGeneration


max_length = 4048


def check_text_data(data):
    """
    Check if the given data is text-based.
    """
    if isinstance(data, str):
        return True
    if isinstance(data, list):
        return all(isinstance(x, str) for x in data)
    return False


def get_pil_from_url(url):
    """
    Downloads and converts an image from a URL to a PIL Image object.
    """
    response = requests.get(url, verify=False, timeout=20)
    image = Image.open(BytesIO(response.content))
    return image.convert("RGB")


def prepare_calibration_data_vision(dataloader, init_steps):
    """
    This function prepares calibration data from a dataloader for a specified number of initialization steps.
    It iterates over the dataloader, fetching batches and storing the relevant data.
    """
    data = []
    print(f"Fetching {init_steps} samples for the initialization...")
    with tqdm(total=init_steps) as pbar:
        for batch in dataloader:
            if len(data) == init_steps:
                break
            if batch:
                pbar.update(1)
                with torch.no_grad():
                    data.append(
                        {
                            "pixel_values": batch["pixel_values"].to("cpu"),
                            "aspect_ratio_ids": batch.data["aspect_ratio_ids"].to("cpu"),
                            "aspect_ratio_mask": batch.data["aspect_ratio_mask"].to("cpu"),
                        }
                    )
    return data


def prepare_dataset_vision(processor, opt_init_steps=50, max_train_samples=1000, file_path="vision_dataset.pickle", save_dataset=True):
    """
    Prepares a vision-text dataset for quantization.
    """

    def collate_fn(example, image_column="image_url", text_column="caption"):
        """
        Preprocesses an example by loading and transforming image and text data.
        Checks if the text data in the example is valid by calling the `check_text_data` function.
        Downloads the image specified by the URL in the image_column by calling the `get_pil_from_url` function.
        If there is any error during the download process, returns None.
        Returns the preprocessed inputs with transformed image and text data.
        """
        assert len(example) == 1
        example = example[0]

        if not check_text_data(example[text_column]):
            raise ValueError("Text data is not valid")

        url = example[image_column]
        try:
            image = get_pil_from_url(url)
            h, w = image.size
            if h == 1 or w == 1:
                return None
        except Exception:
            return None
        inputs = processor(
            text="<|image|><|begin_of_text|> Please describe image content based on information: " + example[text_column],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        if inputs["input_ids"].shape[1] > max_length:
            return None
        return inputs

    if not Path(file_path).exists():
        dataset = load_dataset("google-research-datasets/conceptual_captions", trust_remote_code=True)
        train_dataset = dataset["train"].shuffle(seed=42)
        dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1)
        calibration_data = prepare_calibration_data_vision(dataloader, opt_init_steps)
        print(f"calibration dataset will be saved in {file_path}")
        with open(file_path, "wb") as f:
            pickle.dump(calibration_data, f)
    else:
        with open(file_path, "rb") as f:
            calibration_data = pickle.load(f)

    return calibration_data


def prepare_calibration_data_llm(dataloader, init_steps, mllm, processor):
    """
    This function prepares calibration data from a dataloader for a specified number of initialization steps.
    It iterates over the dataloader, fetching batches and storing the relevant data.
    """
    data = []

    print(f"Fetching {init_steps} samples for the initialization...")
    with tqdm(total=init_steps) as pbar:
        for batch in dataloader:
            if len(data) == init_steps:
                break
            if batch:
                pbar.update(1)
                with torch.no_grad():
                    cache_position = np.cumsum(batch.data["attention_mask"].to("cpu"), axis=1) - 1
                    cache_position[batch.data["attention_mask"] == 0] = 1

                    vision_input = {
                        "pixel_values": batch["pixel_values"].to("cpu"),
                        "aspect_ratio_ids": batch.data["aspect_ratio_ids"].to("cpu"),
                        "aspect_ratio_mask": batch.data["aspect_ratio_mask"].to("cpu"),
                        "cross_attention_mask": batch.data["cross_attention_mask"].to("cpu"),
                        "cache_position": cache_position[0, :],
                    }

                    cross_attention_states = mllm.prepare_vision_outputs(**vision_input)
                    res = {"input_ids": batch.data["input_ids"].to("cpu"), "attention_mask": batch.data["attention_mask"].to("cpu"), **cross_attention_states}
                    position_ids = np.cumsum(res["attention_mask"], axis=1) - 1
                    position_ids[res["attention_mask"] == 0] = 1
                    res["position_ids"] = position_ids

                    res = mllm.prepare_llm_inputs(**res)
                    data.append(res)
    return data


def prepare_dataset_llm(mllm_id, opt_init_steps=50, max_train_samples=1000, file_path="llm_dataset.pickle", save_dataset=False):
    """
    Prepares a vision-text dataset for quantization.
    """

    if Path(file_path).exists():
        print(f"callibration dataset will be loaded from {file_path}")
        with open(file_path, "rb") as f:
            calibration_data = pickle.load(f)
        return calibration_data

    mllm = OVMLlamaForConditionalGeneration(mllm_id, slice_lm_head=False)
    processor = AutoProcessor.from_pretrained(mllm_id)

    def collate_fn(example, image_column="image_url", text_column="caption"):
        """
        Preprocesses an example by loading and transforming image and text data.
        Checks if the text data in the example is valid by calling the `check_text_data` function.
        Downloads the image specified by the URL in the image_column by calling the `get_pil_from_url` function.
        If there is any error during the download process, returns None.
        Returns the preprocessed inputs with transformed image and text data.
        """
        assert len(example) == 1
        example = example[0]

        if not check_text_data(example[text_column]):
            raise ValueError("Text data is not valid")

        url = example[image_column]
        try:
            image = get_pil_from_url(url)
            h, w = image.size
            if h == 1 or w == 1:
                return None
        except Exception:
            return None
        inputs = processor(
            text="<|image|><|begin_of_text|> Please describe image content based on information: " + example[text_column],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        if inputs["input_ids"].shape[1] > max_length:
            return None
        return inputs

    dataset = load_dataset("google-research-datasets/conceptual_captions", trust_remote_code=True)
    train_dataset = dataset["train"].shuffle(seed=42)
    dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1)
    calibration_data = prepare_calibration_data_llm(dataloader, opt_init_steps, mllm, processor)

    if save_dataset:
        with open(file_path, "wb") as f:
            print(f"calibration data will be saved into {file_path}")
            pickle.dump(calibration_data, f)

    del mllm
    gc.collect()

    return calibration_data
