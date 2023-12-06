from analog.paddle import analog_paddle
from analog.yolo import analog_yolo
import argparse
import cv2
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    parser.add_argument('-i', '--input', required=True, type=str,
                      help='Required. Path to an image file.')
    parser.add_argument('-c', '--config',  required=True, type=str,
                      help='Required. config file path')
    parser.add_argument('-t', '--task',  required=True, default='analog', type=str,
                      help='Required. mode of meter reader, digital or analog')
    args = parser.parse_args()
    
    output_dir = os.path.abspath(os.path.dirname(args.input))
    with open(args.config) as f:
        config = json.load(f)
    if len(config["model_config"]["detector"]["model_shape"]) == 1:
        meter_reader = analog_yolo(config, output_dir)
    else:
        meter_reader = analog_paddle(config, output_dir)
    image = cv2.imread(args.input)
    det_resutls = meter_reader.detect(image)
    seg_resutls = meter_reader.segment(det_resutls)
    post_resutls = meter_reader.postprocess(seg_resutls)
    meter_reader.reading(post_resutls, image)
    print(f"result images saved to \"{output_dir}\".")