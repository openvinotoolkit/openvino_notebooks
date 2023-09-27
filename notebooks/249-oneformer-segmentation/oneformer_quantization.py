import warnings
from collections import defaultdict
from pathlib import Path
import sys

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers.models.oneformer.modeling_oneformer import OneFormerForUniversalSegmentationOutput
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from PIL import ImageOps

import nncf
from zipfile import ZipFile
import cv2
import torch.utils.data as data

import openvino

sys.path.append("../utils")
from notebook_utils import download_file

from datasets import load_dataset
import evaluate

DEVICE = 'AUTO'

TASK = "semantic"
# TASK = "instance"
# TASK = "panoptic"

core = openvino.Core()

OUTPUT_NAMES = ['class_queries_logits', 'masks_queries_logits']

SHAPE = (800, 800)


processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
model = OneFormerForUniversalSegmentation.from_pretrained(
    "shi-labs/oneformer_coco_swin_large",
)
id2label = model.config.id2label


class Visualizer:
    @staticmethod
    def extract_legend(handles):
        fig = plt.figure()
        fig.legend(handles=handles, ncol=len(handles) // 20 + 1, loc='center')
        fig.tight_layout()
        return fig

    @staticmethod
    def predicted_semantic_map_to_figure(predicted_map):
        segmentation = predicted_map[0]
        # get the used color map
        viridis = plt.get_cmap('viridis', torch.max(segmentation))
        # get all the unique numbers
        labels_ids = torch.unique(segmentation).tolist()
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        ax.set_axis_off()
        handles = []
        for label_id in labels_ids:
            label = id2label[label_id]
            color = viridis(label_id)
            handles.append(mpatches.Patch(color=color, label=label))
        fig_legend = Visualizer.extract_legend(handles=handles)
        fig.tight_layout()
        return fig, fig_legend

    @staticmethod
    def predicted_instance_map_to_figure(predicted_map):
        segmentation = predicted_map[0]['segmentation']
        segments_info = predicted_map[0]['segments_info']
        # get the used color map
        viridis = plt.get_cmap('viridis', torch.max(segmentation))
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        ax.set_axis_off()
        instances_counter = defaultdict(int)
        handles = []
        # for each segment, draw its legend
        for segment in segments_info:
            segment_id = segment['id']
            segment_label_id = segment['label_id']
            segment_label = id2label[segment_label_id]
            label = f"{segment_label}-{instances_counter[segment_label_id]}"
            instances_counter[segment_label_id] += 1
            color = viridis(segment_id)
            handles.append(mpatches.Patch(color=color, label=label))

        fig_legend = Visualizer.extract_legend(handles)
        fig.tight_layout()
        return fig, fig_legend

    @staticmethod
    def predicted_panoptic_map_to_figure(predicted_map):
        segmentation = predicted_map[0]['segmentation']
        segments_info = predicted_map[0]['segments_info']
        # get the used color map
        viridis = plt.get_cmap('viridis', torch.max(segmentation))
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        ax.set_axis_off()
        instances_counter = defaultdict(int)
        handles = []
        # for each segment, draw its legend
        for segment in segments_info:
            segment_id = segment['id']
            segment_label_id = segment['label_id']
            segment_label = id2label[segment_label_id]
            label = f"{segment_label}-{instances_counter[segment_label_id]}"
            instances_counter[segment_label_id] += 1
            color = viridis(segment_id)
            handles.append(mpatches.Patch(color=color, label=label))

        fig_legend = Visualizer.extract_legend(handles)
        fig.tight_layout()
        return fig, fig_legend


def convert_model(model):
    task_seq_length = processor.task_seq_length
    dummy_input = {
        "pixel_values": torch.randn(1, 3, *SHAPE),
        "task_inputs": torch.randn(1, task_seq_length),
        # "pixel_mask": torch.randn(1, *SHAPE),
    }
    model.config.torchscript = True

    if not IR_PATH.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = openvino.convert_model(model, example_input=dummy_input)
        openvino.save_model(model, IR_PATH, compress_to_fp16=False)


def prepare_inputs(image: Image.Image, task: str):
    """Convert image to model input"""
    image = ImageOps.pad(image, SHAPE)
    inputs = processor(image, [task], return_tensors="pt")
    converted = {
        'pixel_values': inputs['pixel_values'],
        'task_inputs': inputs['task_inputs']
    }
    return converted


def process_output(d):
    """Convert OpenVINO model output to HuggingFace representation for visualization"""
    hf_kwargs = {
        output_name: torch.tensor(d[output_name]) for output_name in OUTPUT_NAMES
    }

    return OneFormerForUniversalSegmentationOutput(**hf_kwargs)


def segment(model, img: Image.Image, task: str):
    """
    Apply segmentation on an image.

    Args:
        img: Input image. It will be resized to 800x800.
        task: String describing the segmentation task. Supported values are: "semantic", "instance" and "panoptic".
    Returns:
        Tuple[Figure, Figure]: Segmentation map and legend charts.
    """
    inputs = prepare_inputs(img, task)
    outputs = model(inputs)
    hf_output = process_output(outputs)
    predicted_map = getattr(processor, f"post_process_{task}_segmentation")(
        hf_output, target_sizes=[img.size[::-1]]
    )
    return getattr(Visualizer, f"predicted_{task}_map_to_figure")(predicted_map)


def run_segmentation(compiled_model, image, task, save_dir: Path):
    result, legend = segment(compiled_model, image, task)
    result.savefig(save_dir / f"result.png", bbox_inches="tight")
    legend.savefig(save_dir / f"legend.png", bbox_inches="tight")


def download_coco128():
    DATA_URL = "https://ultralytics.com/assets/coco128.zip"
    OUT_DIR = Path('.')

    # download_file(DATA_URL, directory=OUT_DIR, show_progress=True)

    if not (OUT_DIR / "coco128/images/train2017").exists():
        with ZipFile('coco128.zip', "r") as zip_ref:
            zip_ref.extractall(OUT_DIR)

    class COCOLoader(data.Dataset):
        def __init__(self, images_path):
            self.images = list(Path(images_path).iterdir())

        def __getitem__(self, index):
            image_path = self.images[index]
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        def __len__(self):
            return len(self.images)

    coco_dataset = COCOLoader(OUT_DIR / 'coco128/images/train2017')

    return coco_dataset


def run_quantization(model, save_dir: Path):
    def transform_fn(image_data):
        """
        Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
        Parameters:
            image_data: image data produced by DataLoader during iteration
        Returns:
            input_tensor: input data in Dict format for model quantization
        """
        image = Image.fromarray(image_data)
        inputs = prepare_inputs(image, TASK)
        return inputs

    save_path = save_dir / "oneformer.xml"
    if save_path.exists():
        quantized_model = core.read_model(save_path)
    else:
        coco_dataset = download_coco128()
        calibration_dataset = nncf.Dataset(coco_dataset, transform_fn)
        print("model quantization started")
        quantized_model = nncf.quantize(model,
                                        calibration_dataset,
                                        model_type=nncf.parameters.ModelType.TRANSFORMER,
                                        preset=nncf.QuantizationPreset.MIXED,
                                        subset_size=len(coco_dataset),
                                        advanced_parameters=nncf.AdvancedQuantizationParameters(
                                            smooth_quant_alpha=0.15,
                                        )
                                        )
        print("model quantization finished")
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        openvino.save_model(quantized_model, save_path)
    compiled_quantized_model = core.compile_model(model=quantized_model, device_name=DEVICE)
    return compiled_quantized_model


# def validate(compiled_model, dataset=None):
#     if dataset is None:
#         dataset = load_dataset("scene_parse_150", name="instance_segmentation", split="validation")[:100]
#
#     print(next(iter(dataset)))
#     exit(0)
#
#     metric = evaluate.load("mean_iou")
#
#     # for data_item in
#     # inputs = prepare_inputs(img, task)
#     # outputs = model(inputs)
#     # hf_output = process_output(outputs)
#     # predicted_map = getattr(processor, f"post_process_{task}_segmentation")(
#     #     hf_output, target_sizes=[img.size[::-1]]
#     # )

IR_PATH = Path("oneformer.xml")

quantization_dir = Path("quantized_models/mixed_sq-0.50")
# IR_PATH = quantization_dir / Path("oneformer.xml")


# convert_model(model)
# exit(0)

model = core.read_model(model=IR_PATH)
# compiled_model = core.compile_model(model=model, device_name=DEVICE)

image = Image.open("sample.jpg")
# run_segmentation(compiled_model, image, TASK, save_dir=Path("./"))

compiled_quantized_model = run_quantization(model, quantization_dir)
run_segmentation(compiled_quantized_model, image, TASK, save_dir=quantization_dir)

# validate(compiled_model)
