import argparse
import os

from ultralytics import YOLO


def convert(det_model_name: str, model_dir: str) -> str:
    """
    Convert YOLO model

    Params:
        det_model_name: name of the YOLO model we want to use
        model_dir: dir to export model
    Returns:
       Path to exported model
    """
    model_path = os.path.join(model_dir, f"{det_model_name}.pt")
    # create a YOLO object detection model
    det_model = YOLO(model_path)

    # export the model to OpenVINO format
    return det_model.export(format="openvino", dynamic=False, half=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                        default="yolov8m", help="Model version to be converted")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")

    args = parser.parse_args()
    convert(args.model_name, args.model_dir)
