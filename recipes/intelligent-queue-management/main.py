import argparse
from pathlib import Path

import app
import convert_and_optimize as convert


def main(args):
    # convert and optimize
    model_path = convert.optimize(args.model_name, Path(args.model_dir), args.quantize, Path(args.data_dir))
    # run
    app.run(args.stream, model_path, args.zones_config_file, args.customers_limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                        default="yolov8m", help="Model version to be converted")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
    parser.add_argument("--quantize", type=bool, default=True, help="Whether the model should be quantized")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to place data in")
    parser.add_argument('--stream', type=str, required=True, help="Path to a video file or the webcam number")
    parser.add_argument('--model_path', type=str, default="model/yolov8m_openvino_int8_model/yolov8m.xml", help="Path to the model")
    parser.add_argument('--zones_config_file', type=str, default="zones.json", help="Path to the zone config file (json)")
    parser.add_argument('--customers_limit', type=int, default=3, help="The maximum number of customers in the queue")

    main(parser.parse_args())
