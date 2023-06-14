import nncf
from typing import Dict
import argparse
import zipfile as zf
from pathlib import Path

from openvino import runtime as ov
from ultralytics import YOLO
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CFG

import utils

DATA_URL = "http://images.cocodataset.org/zips/val2017.zip"
LABELS_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"
CFG_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/datasets/coco.yaml"


def load_model(det_model_name: str, model_dir: Path) -> YOLO:
    """
    Load YOLO model

    Params:
        det_model_name: name of the YOLO model we want to use
        model_dir: dir to export model
    Returns:
       YOLO model
    """
    model_path = model_dir / f"{det_model_name}.pt"
    # create a YOLO object detection model
    return YOLO(model_path)


def convert(det_model_name: str, model_dir: Path) -> Path:
    """
    Convert YOLO model

    Params:
        det_model_name: name of the YOLO model we want to use
        model_dir: dir to export model
    Returns:
       Path to exported model
    """
    det_model = load_model(det_model_name, model_dir)

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


def prepare_data(data_dir: Path, det_model_name: str, model_dir: Path) -> BaseValidator:
    """
    Download COCO dataset and create data loader

    Params:
        data_dir: dir to download data
        det_model_name: name of the YOLO model we want to use
        model_dir: dir to export model
    """
    download_data(data_dir)

    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = str(data_dir / "coco.yaml")

    det_model = load_model(det_model_name, model_dir)

    det_validator = det_model.ValidatorClass(args=args)
    det_validator.data = check_det_dataset(args.data)
    return det_validator


def quantize(data_dir: Path, converted_model_path: Path, det_model_name: str, model_dir: Path) -> Path:
    """
    Quantize converted (IR) model

    Params:
        data_dir: dir to download data
        converted_model_path: path to converted model (IR)
        det_model_name: name of the YOLO model we want to use
        model_dir: dir to export model
    Returns:
       Path to quantized model
    """
    int8_model_path = Path(str(converted_model_path).replace("openvino", "openvino_int8"))

    if not int8_model_path.exists():
        core = ov.Core()

        validator = prepare_data(data_dir, det_model_name, model_dir)
        data_loader = validator.get_dataloader(data_dir / "coco", batch_size=1)

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
                input_tensor: Input data for quantization
            """
            input_tensor = validator.preprocess(data_item)['img'].numpy()
            return input_tensor

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
        model_path = quantize(data_dir, model_path, det_model_name, model_dir)

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
