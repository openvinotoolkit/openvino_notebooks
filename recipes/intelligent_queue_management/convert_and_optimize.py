import argparse
import zipfile as zf
from pathlib import Path
from typing import Dict

import nncf
import numpy as np
from openvino import runtime as ov
from torch.utils.data import DataLoader
from torchvision import datasets
from ultralytics import YOLO
from ultralytics.yolo.data import augment

import utils

DATA_URL = "http://images.cocodataset.org/zips/val2017.zip"
LABELS_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"
CFG_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/ed25db94268fe0e65030f0ddc55e6180e39ffe81/ultralytics/cfg/datasets/coco.yaml"


def convert(det_model_name: str, model_dir: Path) -> Path:
    """
    Convert YOLO model

    Params:
        det_model_name: name of the YOLO model we want to use
        model_dir: dir to export model
    Returns:
       Path to exported model
    """
    model_path = model_dir / f"{det_model_name}.pt"
    # create a YOLO object detection model
    det_model = YOLO(model_path)

    # export the model to OpenVINO format
    output_path = det_model.export(format="openvino", dynamic=False, half=True)
    return Path(output_path) / f"{det_model_name}.xml"


def download_data(data_dir: Path) -> None:
    """
    Download COCO dataset

    Params:
        data_dir: dir to download data
    """
    utils.download_file(DATA_URL, directory=data_dir, show_progress=True)
    utils.download_file(LABELS_URL, directory=data_dir, show_progress=True)
    utils.download_file(CFG_URL, directory=data_dir, show_progress=True)

    if not (data_dir / "coco/labels").exists():
        with zf.ZipFile(data_dir / 'coco2017labels-segments.zip', "r") as zip_ref:
            zip_ref.extractall(data_dir)
        with zf.ZipFile(data_dir / 'val2017.zip', "r") as zip_ref:
            zip_ref.extractall(data_dir / 'coco/images')


def prepare_data(data_dir: Path) -> DataLoader:
    """
    Download COCO dataset and create data loader

    Params:
        data_dir: dir to download data
    Returns:
        PyTorch data loader
    """
    download_data(data_dir)

    # create the COCO validation dataset
    coco_dataset = datasets.CocoDetection(data_dir / "coco/images/val2017",
                                          annFile=data_dir / "coco/annotations/instances_val2017.json",
                                          transform=augment.Compose([lambda x: np.array(x), augment.ClassifyLetterBox(), augment.ToTensor()]))

    # get the loader with batch size 1
    return DataLoader(coco_dataset, batch_size=1, shuffle=True)


def quantize(data_dir: Path, converted_model_path: Path) -> Path:
    """
    Quantize converted (IR) model

    Params:
        data_dir: dir to download data
        converted_model_path: path to converted model (IR)
    Returns:
       Path to quantized model
    """
    int8_model_path = Path(str(converted_model_path).replace("openvino", "openvino_int8"))

    if not int8_model_path.exists():
        core = ov.Core()

        data_loader = prepare_data(data_dir)

        ov_model = core.read_model(converted_model_path)

        ignored_scope = nncf.IgnoredScope(
            types=["Multiply", "Subtract", "Sigmoid"],  # ignore operations
            names=[
                "/model.22/dfl/conv/Conv",  # in the post-processing subgraph
                "/model.22/Add",
                "/model.22/Add_1"
            ]
        )

        # transformation function
        def transform_fn(data_item: Dict):
            """
            Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
            Parameters:
               data_item: Dict with data item produced by DataLoader during iteration
            Returns:
                Input data for quantization
            """
            return data_item[0]

        quantization_dataset = nncf.Dataset(data_loader, transform_fn)

        # quantize
        quantized_model = nncf.quantize(ov_model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED,
                                        ignored_scope=ignored_scope)
        # save to disk
        ov.serialize(quantized_model, str(int8_model_path))

    return int8_model_path


def optimize(det_model_name: str, model_dir: Path, quant: bool, data_dir: Path) -> Path:
    """
    Convert YOLO model

    Params:
        det_model_name: name of the YOLO model we want to use
        model_dir: dir to export model
    Returns:
       Path to exported model
    """
    model_path = convert(det_model_name, model_dir)
    if quant:
        model_path = quantize(data_dir, model_path)

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                        default="yolov8m", help="Model version to be converted")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to place data in")
    parser.add_argument("--quantize", type=bool, default=True, help="Whether the model should be quantized")

    args = parser.parse_args()
    optimize(args.model_name, Path(args.model_dir), args.quantize, Path(args.data_dir))
