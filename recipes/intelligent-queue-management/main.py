import argparse
import pathlib

import app
import convert_and_optimize as convert


def main(args):
    model_dir = convert.convert(args.model_name, args.model_dir)
    model_path = pathlib.Path(model_dir) / f"{args.model_name}.xml"

    app.run(args.stream, model_path, args.zones_config_file, args.customers_limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                        default="yolov8m", help="Model version to be converted")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
    parser.add_argument('--stream', type=str, required=True, help="Path to a video file or the webcam number")
    parser.add_argument('--model_path', type=str, default="model/yolov8m_openvino_int8_model/yolov8m.xml", help="Path to the model")
    parser.add_argument('--zones_config_file', type=str, default="zones.json", help="Path to the zone config file (json)")
    parser.add_argument('--customers_limit', type=int, default=3, help="The maximum number of customers in the queue")

    main(parser.parse_args())
