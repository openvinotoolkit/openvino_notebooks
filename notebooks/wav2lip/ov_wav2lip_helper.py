import numpy as np
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
import torch

import openvino as ov

from notebook_utils import download_file
from Wav2Lip.face_detection.detection.sfd.net_s3fd import s3fd
from Wav2Lip.models import Wav2Lip


def _load(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    model.load_state_dict(new_s)

    return model.eval()


def download_and_convert_models(ov_face_detection_model_path, ov_wav2lip_model_path):
    models_urls = {"s3fd": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"}
    path_to_detector = "checkpoints/face_detection.pth"
    # Convert Face Detection Model
    print("Convert Face Detection Model ...")
    if not os.path.isfile(path_to_detector):
        download_file(models_urls["s3fd"])
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        os.replace("s3fd-619a316812.pth", path_to_detector)
    model_weights = torch.load(path_to_detector)

    face_detector = s3fd()
    face_detector.load_state_dict(model_weights)

    if not ov_face_detection_model_path.exists():
        face_detection_dummy_inputs = torch.FloatTensor(np.random.rand(1, 3, 768, 576))
        face_detection_ov_model = ov.convert_model(face_detector, example_input=face_detection_dummy_inputs)
        ov.save_model(face_detection_ov_model, ov_face_detection_model_path)
    print("Converted face detection OpenVINO model: ", ov_face_detection_model_path)

    print("Convert Wav2Lip Model ...")
    path_to_wav2lip = hf_hub_download(repo_id="numz/wav2lip_studio", filename="Wav2lip/wav2lip.pth", local_dir="checkpoints")
    wav2lip = load_model(path_to_wav2lip)
    img_batch = torch.FloatTensor(np.random.rand(123, 6, 96, 96))
    mel_batch = torch.FloatTensor(np.random.rand(123, 1, 80, 16))

    if not ov_wav2lip_model_path.exists():
        example_inputs = {"audio_sequences": mel_batch, "face_sequences": img_batch}
        wav2lip_ov_model = ov.convert_model(wav2lip, example_input=example_inputs)
        ov.save_model(wav2lip_ov_model, ov_wav2lip_model_path)
    print("Converted face detection OpenVINO model: ", ov_wav2lip_model_path)
