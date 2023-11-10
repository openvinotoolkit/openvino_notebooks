import argparse
from pathlib import Path

import openvino as ov
import torch
from bark.generation import load_model
from torch import nn


# Define the TextEncoderModel class
class TextEncoderModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, idx, past_kv=None):
        return self.encoder(idx, merge_context=True, past_kv=past_kv, use_cache=True)


# Define the CoarseEncoderModel class
class CoarseEncoderModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, idx, past_kv=None):
        return self.encoder(idx, past_kv=past_kv, use_cache=True)


# Define the FineModel class
class FineModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pred_idx, idx):
        b, t, codes = idx.size()
        pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0)
        tok_embs = [
            wte(idx[:, :, i]).unsqueeze(-1)
            for i, wte in enumerate(self.model.transformer.wtes)
        ]
        tok_emb = torch.cat(tok_embs, dim=-1)
        pos_emb = self.model.transformer.wpe(pos)
        x = tok_emb[:, :, :, : pred_idx + 1].sum(dim=-1)
        x = self.model.transformer.drop(x + pos_emb)
        for block in self.model.transformer.h:
            x = block(x)
        x = self.model.transformer.ln_f(x)
        return x


# Function to download and convert the text encoder model
def download_and_convert_text_encoder(models_dir: Path) -> None:
    """
    Downloads and converts the text encoder model to OpenVINO format.
    
    This function checks for the existence of text encoder model files and, if not found,
    it loads the model using load_model function. It then converts the PyTorch model to an
    OpenVINO model and saves it in IR format.

    Parameters:
        models_dir (Path): The directory where the OpenVINO model files will be saved.
    """
    text_model_dir = models_dir / "text_encoder"
    text_model_dir.mkdir(exist_ok=True)
    text_encoder_path1 = text_model_dir / "bark_text_encoder_1.xml"
    text_encoder_path0 = text_model_dir / "bark_text_encoder_0.xml"

    if not text_encoder_path0.exists() or not text_encoder_path1.exists():
        text_encoder = load_model(
            model_type="text", use_gpu=False, force_reload=False
        )
        text_encoder_model = TextEncoderModel(text_encoder["model"])
        ov_model = ov.convert_model(
            text_encoder_model, example_input=torch.ones((1, 513), dtype=torch.int64)
        )
        ov.save_model(ov_model, text_encoder_path0)
        logits, kv_cache = text_encoder_model(torch.ones((1, 513), dtype=torch.int64))
        ov_model = ov.convert_model(
            text_encoder_model,
            example_input=(torch.ones((1, 1), dtype=torch.int64), kv_cache),
        )
        ov.save_model(ov_model, text_encoder_path1)
  

# Function to download and convert the coarse encoder model
def download_and_convert_coarse_encoder(models_dir: Path) -> None:
    """
    Downloads and converts the coarse encoder model to OpenVINO format.

    This function checks for the existence of the coarse encoder model file and, if not present,
    it loads the model with the specified size variant. It then exports the model to OpenVINO
    format and saves the converted model to disk.

    Parameters:
        models_dir (Path): The directory where the OpenVINO model files will be saved.
    """
    coarse_model_dir = models_dir / "coarse_model"
    coarse_model_dir.mkdir(exist_ok=True)
    coarse_encoder_path = coarse_model_dir / "bark_coarse_encoder.xml"
    
    if not coarse_encoder_path.exists():
        coarse_model = load_model(
            model_type="coarse", use_gpu=False, force_reload=False
        )
        coarse_encoder_exportable = CoarseEncoderModel(coarse_model)
        logits, kv_cache = coarse_encoder_exportable(torch.ones((1, 886), dtype=torch.int64))
        ov_model = ov.convert_model(
            coarse_encoder_exportable,
            example_input=(torch.ones((1, 1), dtype=torch.int64), kv_cache),
        )
        ov.save_model(ov_model, coarse_encoder_path)


# Function to download and convert the fine model
def download_and_convert_fine_model(models_dir: Path) -> None:
    """
    Downloads and converts the fine model to OpenVINO format.

    This function checks for the existence of the fine model file and, if not available,
    it loads the fine model. It then converts the fine feature extractor and each language
    model head to OpenVINO format, saving them to the specified directory.

    Parameters:
        models_dir (Path): The directory where the OpenVINO model files will be saved.
    """
    fine_model_dir = models_dir / "fine_model"
    fine_model_dir.mkdir(exist_ok=True)
    fine_feature_extractor_path = fine_model_dir / "bark_fine_feature_extractor.xml"
    if not fine_feature_extractor_path.exists():
        fine_model = load_model(model_type="fine", use_gpu=False, force_reload=False)
        fine_feature_extractor = FineModel(fine_model)
        feature_extractor_out = fine_feature_extractor(
            3, torch.zeros((1, 1024, 8), dtype=torch.int32)
        )
        ov_model = ov.convert_model(
            fine_feature_extractor,
            example_input=(
                torch.ones(1, dtype=torch.long),
                torch.zeros((1, 1024, 8), dtype=torch.long),
            ),
        )
        ov.save_model(ov_model, fine_feature_extractor_path)
        for i, lm_head in enumerate(fine_model.lm_heads):
            lm_head_model = ov.convert_model(
                lm_head, example_input=feature_extractor_out
            )
            ov.save_model(
                lm_head_model,
                fine_model_dir / f"bark_fine_lm_{i}.xml",
            )


def convert_bark(use_small: bool) -> None:
    """
    This function orchestrates the process of downloading and converting the text encoder,
    coarse encoder, and fine model based on the specified model size variant. 

    Parameters:
        use_small (bool): Flag indicating whether to download and convert the smaller variants
                          of the models. This will also be reflected in the directory names.
    """
    model_suffix = "_small" if use_small else ""
    models_dir = Path(f"model/TTS_bark{model_suffix}")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    download_and_convert_text_encoder(models_dir)
    download_and_convert_coarse_encoder(models_dir)
    download_and_convert_fine_model(models_dir)
    
    print("All models have been downloaded and converted successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert models")
    parser.add_argument("--use_small_models", default=False, action="store_true", help="Use smaller model variants")
    args = parser.parse_args()

    convert_bark(args.use_small_models)
