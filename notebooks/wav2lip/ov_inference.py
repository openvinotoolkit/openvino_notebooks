from glob import glob
from enum import Enum
import math
import subprocess

import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from Wav2Lip import audio
import openvino as ov


device = "cpu"


def bboxlog(x1, y1, x2, y2, axc, ayc, aww, ahh):
    xc, yc, ww, hh = (x2 + x1) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1
    dx, dy = (xc - axc) / aww, (yc - ayc) / ahh
    dw, dh = math.log(ww / aww), math.log(hh / ahh)
    return dx, dy, dw, dh


def bboxloginv(dx, dy, dw, dh, axc, ayc, aww, ahh):
    xc, yc = dx * aww + axc, dy * ahh + ayc
    ww, hh = math.exp(dw) * aww, math.exp(dh) * ahh
    x1, x2, y1, y2 = xc - ww / 2, xc + ww / 2, yc - hh / 2, yc + hh / 2
    return x1, y1, x2, y2


def nms(dets, thresh):
    if 0 == len(dets):
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= variances[0] * priors[:, 2:]
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def batch_decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:], priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1])), 2)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T :]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def detect(net, img, device):
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,) + img.shape)

    img = torch.from_numpy(img).float().to(device)
    BB, CC, HH, WW = img.size()

    results = net({"x": img})
    olist = [torch.Tensor(results[i]) for i in range(12)]

    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)
    olist = [oelem.data.cpu() for oelem in olist]
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.size()  # feature map size
        stride = 2 ** (i + 2)  # 4,8,16,32,64,128
        anchor = stride * 4
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            score = ocls[0, 1, hindex, windex]
            loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
            priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
            variances = [0.1, 0.2]
            box = decode(loc, priors, variances)
            x1, y1, x2, y2 = box[0] * 1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append([x1, y1, x2, y2, score])
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, 5))

    return bboxlist


def batch_detect(net, imgs, device):
    imgs = imgs - np.array([104, 117, 123])
    imgs = imgs.transpose(0, 3, 1, 2)

    imgs = torch.from_numpy(imgs).float().to(device)
    BB, CC, HH, WW = imgs.size()

    results = net({"x": imgs.numpy()})
    olist = [torch.Tensor(results[i]) for i in range(12)]

    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)
        # olist[i * 2] = (olist[i * 2], dim=1)
    olist = [oelem.data.cpu() for oelem in olist]
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.size()  # feature map size
        stride = 2 ** (i + 2)  # 4,8,16,32,64,128
        anchor = stride * 4
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            score = ocls[:, 1, hindex, windex]
            loc = oreg[:, :, hindex, windex].contiguous().view(BB, 1, 4)
            priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]).view(1, 1, 4)
            variances = [0.1, 0.2]
            box = batch_decode(loc, priors, variances)
            box = box[:, 0] * 1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append(torch.cat([box, score.unsqueeze(1)], 1).cpu().numpy())
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, BB, 5))

    return bboxlist


def flip_detect(net, img, device):
    img = cv2.flip(img, 1)
    b = detect(net, img, device)

    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = img.shape[1] - b[:, 2]
    bboxlist[:, 1] = b[:, 1]
    bboxlist[:, 2] = img.shape[1] - b[:, 0]
    bboxlist[:, 3] = b[:, 3]
    bboxlist[:, 4] = b[:, 4]
    return bboxlist


def pts_to_bb(pts):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    return np.array([min_x, min_y, max_x, max_y])


class OVFaceDetector(object):
    """An abstract class representing a face detector.

    Any other face detection implementation must subclass it. All subclasses
    must implement ``detect_from_image``, that return a list of detected
    bounding boxes. Optionally, for speed considerations detect from path is
    recommended.
    """

    def __init__(self, device, verbose):
        self.device = device
        self.verbose = verbose

    def detect_from_image(self, tensor_or_path):
        """Detects faces in a given image.

        This function detects the faces present in a provided BGR(usually)
        image. The input can be either the image itself or the path to it.

        Arguments:
            tensor_or_path {numpy.ndarray, torch.tensor or string} -- the path
            to an image or the image itself.

        Example::

            >>> path_to_image = 'data/image_01.jpg'
            ...   detected_faces = detect_from_image(path_to_image)
            [A list of bounding boxes (x1, y1, x2, y2)]
            >>> image = cv2.imread(path_to_image)
            ...   detected_faces = detect_from_image(image)
            [A list of bounding boxes (x1, y1, x2, y2)]

        """
        raise NotImplementedError

    def detect_from_directory(self, path, extensions=[".jpg", ".png"], recursive=False, show_progress_bar=True):
        """Detects faces from all the images present in a given directory.

        Arguments:
            path {string} -- a string containing a path that points to the folder containing the images

        Keyword Arguments:
            extensions {list} -- list of string containing the extensions to be
            consider in the following format: ``.extension_name`` (default:
            {['.jpg', '.png']}) recursive {bool} -- option wherever to scan the
            folder recursively (default: {False}) show_progress_bar {bool} --
            display a progressbar (default: {True})

        Example:
        >>> directory = 'data'
        ...   detected_faces = detect_from_directory(directory)
        {A dictionary of [lists containing bounding boxes(x1, y1, x2, y2)]}

        """
        if self.verbose:
            logger = logging.getLogger(__name__)

        if len(extensions) == 0:
            if self.verbose:
                logger.error("Expected at list one extension, but none was received.")
            raise ValueError

        if self.verbose:
            logger.info("Constructing the list of images.")
        additional_pattern = "/**/*" if recursive else "/*"
        files = []
        for extension in extensions:
            files.extend(glob.glob(path + additional_pattern + extension, recursive=recursive))

        if self.verbose:
            logger.info("Finished searching for images. %s images found", len(files))
            logger.info("Preparing to run the detection.")

        predictions = {}
        for image_path in tqdm(files, disable=not show_progress_bar):
            if self.verbose:
                logger.info("Running the face detector on image: %s", image_path)
            predictions[image_path] = self.detect_from_image(image_path)

        if self.verbose:
            logger.info("The detector was successfully run on all %s images", len(files))

        return predictions

    @property
    def reference_scale(self):
        raise NotImplementedError

    @property
    def reference_x_shift(self):
        raise NotImplementedError

    @property
    def reference_y_shift(self):
        raise NotImplementedError

    @staticmethod
    def tensor_or_path_to_ndarray(tensor_or_path, rgb=True):
        """Convert path (represented as a string) or torch.tensor to a numpy.ndarray

        Arguments:
            tensor_or_path {numpy.ndarray, torch.tensor or string} -- path to the image, or the image itself
        """
        if isinstance(tensor_or_path, str):
            return cv2.imread(tensor_or_path) if not rgb else cv2.imread(tensor_or_path)[..., ::-1]
        elif torch.is_tensor(tensor_or_path):
            # Call cpu in case its coming from cuda
            return tensor_or_path.cpu().numpy()[..., ::-1].copy() if not rgb else tensor_or_path.cpu().numpy()
        elif isinstance(tensor_or_path, np.ndarray):
            return tensor_or_path[..., ::-1].copy() if not rgb else tensor_or_path
        else:
            raise TypeError


class OVSFDDetector(OVFaceDetector):
    def __init__(self, device, path_to_detector="models/face_detection.xml", verbose=False):
        super(OVSFDDetector, self).__init__(device, verbose)

        core = ov.Core()
        self.face_detector = core.compile_model(path_to_detector, self.device)

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image, device="cpu")
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    def detect_from_batch(self, images):
        bboxlists = batch_detect(self.face_detector, images, device="cpu")
        keeps = [nms(bboxlists[:, i, :], 0.3) for i in range(bboxlists.shape[1])]
        bboxlists = [bboxlists[keep, i, :] for i, keep in enumerate(keeps)]
        bboxlists = [[x for x in bboxlist if x[-1] > 0.5] for bboxlist in bboxlists]

        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """

    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


class OVFaceAlignment:
    def __init__(
        self, landmarks_type, network_size=NetworkSize.LARGE, device="CPU", flip_input=False, verbose=False, path_to_detector="models/face_detection.xml"
    ):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        self.face_detector = OVSFDDetector(device=device, path_to_detector=path_to_detector, verbose=verbose)

    def get_detections_for_batch(self, images):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)

            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))

        return results


def face_detect_ov(images, device, face_det_batch_size, pads, nosmooth, path_to_detector):
    detector = OVFaceAlignment(LandmarksType._2D, flip_input=False, device=device, path_to_detector=path_to_detector)

    batch_size = face_det_batch_size

    print("face_detect_ov images[0].shape: ", images[0].shape)
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i : i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError("Image too big to run face detection on GPU. Please use the --resize_factor argument")
            batch_size //= 2
            print("Recovering from OOM error; New batch size: {}".format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            # check this frame where the face was not detected.
            cv2.imwrite("temp/faulty_frame.jpg", image)
            raise ValueError("Face not detected! Ensure the video contains a face in all the frames.")

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results


def datagen(frames, mels, box, static, face_det_batch_size, pads, nosmooth, img_size, wav2lip_batch_size, path_to_detector):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if box[0] == -1:
        if not static:
            # BGR2RGB for CNN face detection
            face_det_results = face_detect_ov(frames, "CPU", face_det_batch_size, pads, nosmooth, path_to_detector)
        else:
            face_det_results = face_detect_ov([frames[0]], "CPU", face_det_batch_size, pads, nosmooth, path_to_detector)
    else:
        print("Using the specified bounding box instead of face detection...")
        y1, y2, x1, x2 = box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size // 2 :] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size // 2 :] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def ov_inference(
    face_path,
    audio_path,
    face_detection_path="models/face_detection.xml",
    wav2lip_path="models/wav2lip.xml",
    inference_device="CPU",
    wav2lip_batch_size=128,
    outfile="results/result_voice.mp4",
    resize_factor=1,
    rotate=False,
    crop=[0, -1, 0, -1],
    mel_step_size=16,
    box=[-1, -1, -1, -1],
    static=False,
    img_size=96,
    face_det_batch_size=16,
    pads=[0, 10, 0, 0],
    nosmooth=False,
):
    print("Reading video frames...")

    video_stream = cv2.VideoCapture(face_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

        if rotate:
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

        y1, y2, x1, x2 = crop
        if x2 == -1:
            x2 = frame.shape[1]
        if y2 == -1:
            y2 = frame.shape[0]

        frame = frame[y1:y2, x1:x2]

        full_frames.append(frame)

    print("Number of frames available for inference: " + str(len(full_frames)))

    core = ov.Core()

    if not audio_path.endswith(".wav"):
        print("Extracting raw audio...")
        command = "ffmpeg -y -i {} -strict -2 {}".format(audio_path, "temp/temp.wav")

        subprocess.call(command, shell=True)
        audio_path = "temp/temp.wav"

    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError("Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again")

    mel_chunks = []
    mel_idx_multiplier = 80.0 / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[: len(mel_chunks)]
    batch_size = wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks, box, static, face_det_batch_size, pads, nosmooth, img_size, wav2lip_batch_size, face_detection_path)
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
        if i == 0:
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            compiled_wav2lip_model = core.compile_model(wav2lip_path, inference_device)
            print("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter("Wav2Lip/temp/result.avi", cv2.VideoWriter_fourcc(*"DIVX"), fps, (frame_w, frame_h))
            pred_ov = compiled_wav2lip_model({"audio_sequences": mel_batch.numpy(), "face_sequences": img_batch.numpy()})[0]
        else:
            img_batch = np.transpose(img_batch, (0, 3, 1, 2))
            mel_batch = np.transpose(mel_batch, (0, 3, 1, 2))
            pred_ov = compiled_wav2lip_model({"audio_sequences": mel_batch, "face_sequences": img_batch})[0]

        pred_ov = compiled_wav2lip_model({"audio_sequences": mel_batch, "face_sequences": img_batch})[0]
        pred_ov = pred_ov.transpose(0, 2, 3, 1) * 255.0
        for p, f, c in zip(pred_ov, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    command = "ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}".format(audio_path, "Wav2Lip/temp/result.avi", outfile)
    subprocess.call(command, shell=True)

    return outfile
