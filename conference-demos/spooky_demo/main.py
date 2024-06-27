import argparse
import collections
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from numpy.lib.stride_tricks import as_strided

from decoder import OpenPoseDecoder

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils


# 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    """
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
    )
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling.
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


# non maximum suppression
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)


# Get poses from results.
def process_results(img, pafs, heatmaps, model, decoder):
    # This processing comes from
    # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
    pooled_heatmaps = np.array(
        [[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
    )
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    # Decode poses.
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(model.output(index=0).partial_shape)
    output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()
    # Multiply coordinates by a scaling factor.
    poses[:, :, :2] *= output_scale
    return poses, scores


default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
                    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (17, 18), (20, 21), (23, 24), (26, 27), (29, 30))


pumpkin_img = cv2.imread("assets/pumpkin.png", cv2.IMREAD_UNCHANGED)


def add_artificial_points(pose, point_score_threshold):
    # neck, bellybutton, ribs
    neck = (pose[5] + pose[6]) / 2
    bellybutton = (pose[11] + pose[12]) / 2
    if neck[2] > point_score_threshold and bellybutton[2] > point_score_threshold:
        rib_1_center = (neck + bellybutton) / 2
        rib_1_left = (pose[5] + bellybutton) / 2
        rib_1_right = (pose[6] + bellybutton) / 2
        rib_2_center = (neck + rib_1_center) / 2
        rib_2_left = (pose[5] + rib_1_left) / 2
        rib_2_right = (pose[6] + rib_1_right) / 2
        rib_3_center = (neck + rib_2_center) / 2
        rib_3_left = (pose[5] + rib_2_left) / 2
        rib_3_right = (pose[6] + rib_2_right) / 2
        rib_4_center = (rib_1_center + rib_2_center) / 2
        rib_4_left = (rib_1_left + rib_2_left) / 2
        rib_4_right = (rib_1_right + rib_2_right) / 2
        new_points = [neck, bellybutton, rib_1_center, rib_1_left, rib_1_right, rib_2_center, rib_2_left, rib_2_right,
                      rib_3_center, rib_3_left, rib_3_right, rib_4_center, rib_4_left, rib_4_right]
        pose = np.vstack([pose, new_points])
    return pose


def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.multiply(img, 0.5)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if poses.size == 0:
        return img

    for pose in poses:
        pose = add_artificial_points(pose, point_score_threshold)

        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]

        out_thickness = img.shape[0] // 100
        if points_scores[5] > point_score_threshold and points_scores[6] > point_score_threshold:
            out_thickness = max(2, abs(points[5, 0] - points[6, 0]) // 15)
        in_thickness = out_thickness // 2

        img_limbs = np.copy(img)
        # Draw limbs.
        for i, j in skeleton:
            if i < len(points_scores) and j < len(points_scores) and points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=(0, 0, 0), thickness=out_thickness, lineType=cv2.LINE_AA)
                cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=(255, 255, 255), thickness=in_thickness, lineType=cv2.LINE_AA)
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img_limbs, tuple(p), 1, color=(0, 0, 0), thickness=2 * out_thickness, lineType=cv2.LINE_AA)
                cv2.circle(img_limbs, tuple(p), 1, color=(255, 255, 255), thickness=2 * in_thickness, lineType=cv2.LINE_AA)

        cv2.addWeighted(img, 0.3, img_limbs, 0.7, 0, dst=img)

        face_size_scale = 2.2
        left_ear = 3
        right_ear = 4
        left_eye = 1
        right_eye = 2
        # if left eye and right eye and left ear or right ear are visible
        if points_scores[left_eye] > point_score_threshold and points_scores[right_eye] > point_score_threshold and (points_scores[left_ear] > point_score_threshold or points_scores[right_ear] > point_score_threshold):
            # visible left ear and right ear
            if points_scores[left_ear] > point_score_threshold and points_scores[right_ear] > point_score_threshold:
                face_width = np.linalg.norm(points[left_ear] - points[right_ear]) * face_size_scale
                face_center = (points[left_ear] + points[right_ear]) // 2
            # visible left ear and right eye
            elif points_scores[left_ear] > point_score_threshold and points_scores[right_eye] > point_score_threshold:
                face_width = np.linalg.norm(points[left_ear] - points[right_eye]) * face_size_scale
                face_center = (points[left_ear] + points[right_eye]) // 2
            # visible right ear and left eye
            elif points_scores[left_eye] > point_score_threshold and points_scores[right_ear] > point_score_threshold:
                face_width = np.linalg.norm(points[left_eye] - points[right_ear]) * face_size_scale
                face_center = (points[left_eye] + points[right_ear]) // 2

            face_width = max(1.0, face_width)
            scale = face_width / pumpkin_img.shape[1]
            pumpkin_face = cv2.resize(pumpkin_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # left-top and right-bottom points
            x1, y1 = face_center[0] - pumpkin_face.shape[1] // 2, face_center[1] - pumpkin_face.shape[0] * 2 // 3
            x2, y2 = face_center[0] + pumpkin_face.shape[1] // 2, face_center[1] + pumpkin_face.shape[0] // 3

            # face image to be overlayed
            face_crop = img[max(0, y1):min(y2, img.shape[0]), max(0, x1):min(x2, img.shape[1])]
            # overlay
            pumpkin_face = pumpkin_face[max(0, -y1):max(0, -y1) + face_crop.shape[0], max(0, -x1):max(0, -x1) + face_crop.shape[1]]
            # alpha channel to blend images
            alpha_pumpkin = pumpkin_face[:, :, 3:4] / 255.0
            alpha_bg = 1.0 - alpha_pumpkin

            # blend images
            face_crop[:] = (alpha_pumpkin * pumpkin_face)[:, :, :3] + alpha_bg * face_crop

    return img


def load_and_compile_model(model_name, precision, device):
    base_model_dir = Path("model")

    model_path = base_model_dir / "intel" / model_name / precision / f"{model_name}.xml"

    if not model_path.exists():
        model_url_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
        utils.download_file(model_url_dir + model_name + '.xml', model_path.name, model_path.parent)
        utils.download_file(model_url_dir + model_name + '.bin', model_path.with_suffix('.bin').name, model_path.parent)

    # Initialize OpenVINO Runtime
    core = ov.Core()
    # Read the network from a file.
    model = core.read_model(model_path)
    # Let the AUTO device decide where to load the model (you can use CPU, GPU as well).
    compiled_model = core.compile_model(model=model, device_name=device, config={"PERFORMANCE_HINT": "LATENCY"})
    return compiled_model


def run_demo(source, model_name, model_precision, device, flip):
    decoder = OpenPoseDecoder()

    compiled_model = load_and_compile_model(model_name, model_precision, device)

    # Get the input and output names of nodes.
    input_layer = compiled_model.input(0)

    # Get the input size.
    height, width = list(input_layer.shape)[2:]

    pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")
    player = None
    try:
        if isinstance(source, str) and source.isnumeric():
            source = int(source)
        # Create a video player to play with target fps.
        player = utils.VideoPlayer(source, flip=flip, fps=30, size=(1280, 720))
        # Start capturing.
        player.start()
        title = "Press ESC to Exit"
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        processing_times = collections.deque()

        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Resize the image and change dims to fit neural network input.
            # (see https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001)
            input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # Create a batch of images (size = 1).
            input_img = input_img.transpose((2,0,1))[np.newaxis, ...]

            # Measure processing time.
            start_time = time.time()
            # Get results.
            results = compiled_model([input_img])
            stop_time = time.time()

            # Draw watermark
            utils.draw_ov_watermark(frame)

            pafs = results[pafs_output_key]
            heatmaps = results[heatmaps_output_key]
            # Get poses from network results.
            poses, scores = process_results(frame, pafs, heatmaps, compiled_model, decoder)

            # Draw poses on a frame.
            frame = draw_poses(frame, poses, 0.25)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # mean processing time [ms]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),
                        cv2.FONT_HERSHEY_COMPLEX, f_width / 1500, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),
                        cv2.FONT_HERSHEY_COMPLEX, f_width / 1500, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(title, frame)
            key = cv2.waitKey(1)
            # escape = 27
            if key == 27:
                break
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default="0", type=str, help="Path to a video file or the webcam number")
    parser.add_argument('--device', default="AUTO", type=str, help="Device to run inference on")
    parser.add_argument("--model_name", type=str, default="human-pose-estimation-0001", help="Pose estimation model to be used")
    parser.add_argument("--model_precision", type=str, default="FP16-INT8", choices=["FP16-INT8", "FP16", "FP32"], help="Pose estimation model precision")
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")

    args = parser.parse_args()
    run_demo(args.stream, args.model_name, args.model_precision, args.device, args.flip)
