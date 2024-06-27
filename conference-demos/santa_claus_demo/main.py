import argparse
import collections
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from openvino.runtime import Core

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils


def download_model(model_name, precision):
    base_model_dir = Path("model")

    model_path = base_model_dir / "intel" / model_name / precision / f"{model_name}.xml"

    if not model_path.exists():
        model_url_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
        utils.download_file(model_url_dir + model_name + '.bin', model_path.with_suffix('.bin').name, model_path.parent)
        utils.download_file(model_url_dir + model_name + '.xml', model_path.name, model_path.parent)

    return model_path


def load_model(model_path, device):
    # Initialize OpenVINO Runtime.
    core = Core()

    # Read the network and corresponding weights from a file.
    model = core.read_model(model=model_path)
    # Compile the model for CPU (you can choose manually CPU, GPU, MYRIAD etc.)
    # or let the engine choose the best available device (AUTO).
    compiled_model = core.compile_model(model=model, device_name=device)

    # Get the input and output nodes.
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    return compiled_model, input_layer, output_layer


def preprocess_images(imgs, width, height):
    result = []
    for img in imgs:
        # Resize the image and change dims to fit neural network input.
        input_img = cv2.resize(src=img, dsize=(width, height), interpolation=cv2.INTER_AREA)
        input_img = input_img.transpose(2, 0, 1)[np.newaxis, ...]
        result.append(input_img)
    return np.array(result)


def process_detection_results(frame, results, thresh=0.8):
    # The size of the original frame.
    h, w = frame.shape[:2]
    # The 'results' variable is a [1, 1, 100, 7] tensor.
    results = results.squeeze()
    boxes = []
    scores = []
    for _, _, score, xmin, ymin, xmax, ymax in results:
        # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        boxes.append(tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h))))
        scores.append(float(score))

    # Apply non-maximum suppression to get rid of many overlapping entities.
    # See https://paperswithcode.com/method/non-maximum-suppression
    # This algorithm returns indices of objects to keep.
    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6)

    # If there are no boxes.
    if len(indices) == 0:
        return []

    # Filter detected objects.
    return [(scores[idx], boxes[idx]) for idx in indices.flatten()]


def process_landmark_results(boxes, results):
    landmarks = []

    for box, result in zip(boxes, results):
        # create a vector of landmarks (5x2)
        result = result.reshape(-1, 2)
        box = box[1]
        # move every landmark according to box origin
        landmarks.append((result * box[2:] + box[:2]).astype(np.int32))

    return landmarks


santa_beard_img = cv2.imread("assets/santa_beard.png", cv2.IMREAD_UNCHANGED)
santa_cap_img = cv2.imread("assets/santa_cap.png", cv2.IMREAD_UNCHANGED)
reindeer_nose_img = cv2.imread("assets/reindeer_nose.png", cv2.IMREAD_UNCHANGED)
reindeer_sunglasses_img = cv2.imread("assets/reindeer_sunglasses.png", cv2.IMREAD_UNCHANGED)
reindeer_antlers_img = cv2.imread("assets/reindeer_antlers.png", cv2.IMREAD_UNCHANGED)


def draw_mask(img, mask_img, center, face_size, scale=1.0, offset_coeffs=(0.5, 0.5)):
    face_width, face_height = face_size

    # scale mask to fit face size
    mask_width = max(1.0, face_width * scale)
    f_scale = mask_width / mask_img.shape[1]
    mask_img = cv2.resize(mask_img, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_AREA)

    x_offset_coeff, y_offset_coeff = offset_coeffs

    # left-top and right-bottom points
    x1, y1 = center[0] - int(mask_img.shape[1] * x_offset_coeff), center[1] - int(mask_img.shape[0] * y_offset_coeff)
    x2, y2 = x1 + mask_img.shape[1], y1 + mask_img.shape[0]

    # if points inside image
    if 0 < x2 < img.shape[1] and 0 < y2 < img.shape[0] or 0 < x1 < img.shape[1] and 0 < y1 < img.shape[1]:
        # face image to be overlayed
        face_crop = img[max(0, y1):min(y2, img.shape[0]), max(0, x1):min(x2, img.shape[1])]
        # overlay
        mask_img = mask_img[max(0, -y1):max(0, -y1) + face_crop.shape[0], max(0, -x1):max(0, -x1) + face_crop.shape[1]]
        # alpha channel to blend images
        alpha_pumpkin = mask_img[:, :, 3:4] / 255.0
        alpha_bg = 1.0 - alpha_pumpkin

        # blend images
        face_crop[:] = (alpha_pumpkin * mask_img)[:, :, :3] + alpha_bg * face_crop


def draw_santa(img, detection):
    (score, box), landmarks, emotion = detection
    # draw beard
    draw_mask(img, santa_beard_img, landmarks[2], box[2:], offset_coeffs=(0.5, -0.05))
    # draw cap
    draw_mask(img, santa_cap_img, np.mean(landmarks[:2], axis=0, dtype=np.int32), box[2:], scale=1.5, offset_coeffs=(0.56, 0.87))


def draw_reindeer(img, landmarks, box):
    # draw nose
    draw_mask(img, reindeer_nose_img, landmarks[2], box[2:], scale=0.25)
    # draw antlers
    draw_mask(img, reindeer_antlers_img, np.mean(landmarks[:2], axis=0, dtype=np.int32), box[2:], scale=1.8, offset_coeffs=(0.5, 1.2))
    # draw sunglasses
    draw_mask(img, reindeer_sunglasses_img, np.mean(landmarks[:2], axis=0, dtype=np.int32), box[2:], offset_coeffs=(0.5, 0.33))


def draw_christmas_masks(frame, detections):
    # sort by face size
    detections = list(sorted(detections, key=lambda x: x[0][1][2] * x[0][1][3]))

    if not detections:
        return frame

    # others are reindeer
    for (score, box), landmarks, emotion in detections[:-1]:
        draw_reindeer(frame, landmarks, box)

        (label_width, label_height), _ = cv2.getTextSize(
            text=emotion_mapping[emotion],
            fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            fontScale=box[2] / 150,
            thickness=1)
        point = np.mean(landmarks[:2], axis=0, dtype=np.int32) - [label_width // 2, 2 * label_height]
        cv2.putText(
            img=frame,
            text=emotion_mapping[emotion],
            org=point,
            fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            fontScale=box[2] / 150,
            color=(0, 0, 196),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # the largest face is santa
    draw_santa(frame, detections[-1])

    return frame


emotion_classes = ["neutral", "happy", "sad", "surprise", "anger"]
emotion_mapping = {"neutral": "Rudolph", "happy": "Cupid", "surprise": "Blitzen", "sad": "Prancer", "anger": "Vixen"}


def run_demo(source, face_detection_model, face_landmarks_model, face_emotions_model, model_precision, device, flip):
    face_detection_model_path = download_model(face_detection_model, model_precision)
    face_landmarks_model_path = download_model(face_landmarks_model, model_precision)
    face_emotions_model_path = download_model(face_emotions_model, model_precision)

    # load face detection model
    fd_model, fd_input, fd_output = load_model(face_detection_model_path, device)
    fd_height, fd_width = list(fd_input.shape)[2:4]

    # load face landmarks model
    fl_model, fl_input, fl_output = load_model(face_landmarks_model_path, device)
    fl_height, fl_width = list(fl_input.shape)[2:4]

    # load emotion classification model
    fe_model, fe_input, fe_output = load_model(face_emotions_model_path, device)
    fe_height, fe_width = list(fe_input.shape)[2:4]

    def detect_faces(img):
        input_img = preprocess_images([img], fd_width, fd_height)[0]
        results = fd_model([input_img])[fd_output]
        return process_detection_results(frame=img, results=results)

    def detect_landmarks(img, boxes):
        # every patch is a face image
        patches = [img[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :] for _, box in boxes]
        patches = preprocess_images(patches, fl_width, fl_height)
        # there are many faces on the image
        results = [fl_model([patch])[fl_output].squeeze() for patch in patches]
        return process_landmark_results(boxes, results)

    def recognize_emotions(img, boxes):
        # every patch is a face image
        patches = [img[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :] for _, box in boxes]
        patches = preprocess_images(patches, fe_width, fe_height)
        # there are many faces on the image
        results = [fe_model([patch])[fe_output].squeeze() for patch in patches]

        if not results:
            return []

        # map result to labels
        labels = list(map(lambda i: emotion_classes[i], np.argmax(results, axis=1)))
        return labels

    player = None
    try:
        if isinstance(source, str) and source.isnumeric():
            source = int(source)
        # Create a video player to play with target fps.
        player = utils.VideoPlayer(source=source, flip=flip, size=(1920, 1080), fps=30)
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

            # Measure processing time.
            start_time = time.time()

            boxes = detect_faces(frame)
            landmarks = detect_landmarks(frame, boxes)
            emotions = recognize_emotions(frame, boxes)
            detections = zip(boxes, landmarks, emotions)

            stop_time = time.time()

            # Draw watermark
            utils.draw_ov_watermark(frame)

            # Draw boxes on a frame.
            frame = draw_christmas_masks(frame, detections)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),
                        cv2.FONT_HERSHEY_COMPLEX, f_width / 1500, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),
                        cv2.FONT_HERSHEY_COMPLEX, f_width / 1500, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(winname=title, mat=frame)
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
    parser.add_argument("--detection_model_name", type=str, default="face-detection-adas-0001", help="Face detection model to be used")
    parser.add_argument("--landmarks_model_name", type=str, default="landmarks-regression-retail-0009", help="Face landmarks regression model to be used")
    parser.add_argument("--emotions_model_name", type=str, default="emotions-recognition-retail-0003", help="Face emotions recognition model to be used")
    parser.add_argument("--model_precision", type=str, default="FP16-INT8", choices=["FP16-INT8", "FP16", "FP32"], help="Pose estimation model precision")
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")

    args = parser.parse_args()
    run_demo(args.stream, args.detection_model_name, args.landmarks_model_name, args.emotions_model_name, args.model_precision, args.device, args.flip)