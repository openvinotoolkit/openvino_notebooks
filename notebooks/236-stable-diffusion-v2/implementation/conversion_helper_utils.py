from pathlib import Path
import gc
import torch
import numpy as np
from openvino.tools.mo import convert_model
from openvino.runtime import serialize


def convert_encoder_onnx(text_encoder: torch.nn.Module, onnx_path:Path):
    """
    Convert Text Encoder model to ONNX. 
    Function accepts pipeline, prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        text_encoder (torch.nn.Module): text encoder PyTorch model
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    if not onnx_path.exists():
        input_ids = torch.ones((1, 77), dtype=torch.long)
        # switch model to inference mode
        text_encoder.eval()

        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            # export model to ONNX format
            torch.onnx._export(
                text_encoder,  # model instance
                input_ids,  # inputs for model tracing
                onnx_path,  # output file for saving result
                input_names=['tokens'],  # model input name for onnx representation
                output_names=['last_hidden_state', 'pooler_out'],  # model output names for onnx representation
                opset_version=14,  # onnx opset version for export,
                onnx_shape_inference=False
            )
        print('Text Encoder successfully converted to ONNX')

        
def convert_unet_onnx(unet:torch.nn.Module, onnx_path:Path, num_channels:int = 4, width:int = 64, height:int = 64):
    """
    Convert Unet model to ONNX, then IR format. 
    Function accepts pipeline, prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        unet (torch.nn.Module): UNet PyTorch model
        onnx_path (Path): File for storing onnx model
        num_channels (int, optional, 4): number of input channels
        width (int, optional, 64): input width
        height (int, optional, 64): input height
    Returns:
        None
    """
    if not onnx_path.exists():
        # prepare inputs
        encoder_hidden_state = torch.ones((2, 77, 1024))
        latents_shape = (2, num_channels, width, height)
        latents = torch.randn(latents_shape)
        t = torch.from_numpy(np.array(1, dtype=np.float32))

        # model size > 2Gb, it will be represented as onnx with external data files, we will store it in separated directory for avoid a lot of files in current directory
        onnx_path.parent.mkdir(exist_ok=True, parents=True)
        unet.eval()

        with torch.no_grad():
            torch.onnx._export(
                unet, 
                (latents, t, encoder_hidden_state), str(onnx_path),
                input_names=['latent_model_input', 't', 'encoder_hidden_states'],
                output_names=['out_sample'],
                onnx_shape_inference=False
            )
        print('U-Net successfully converted to ONNX')


def convert_vae_encoder_onnx(vae: torch.nn.Module, onnx_path: Path, width:int = 512, height:int = 512):
    """
    Convert VAE model to ONNX, then IR format. 
    Function accepts pipeline, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        vae (torch.nn.Module): VAE PyTorch model
        onnx_path (Path): File for storing onnx model
        width (int, optional, 512): input width
        height (int, optional, 512): input height
    Returns:
        None
    """
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, image):
            h = self.vae.encoder(image)
            moments = self.vae.quant_conv(h)
            return moments

    if not onnx_path.exists():
        vae_encoder = VAEEncoderWrapper(vae)
        vae_encoder.eval()
        image = torch.zeros((1, 3, width, height))
        with torch.no_grad():
            torch.onnx.export(vae_encoder, image, onnx_path, input_names=[
                              'init_image'], output_names=['image_latent'])
        print('VAE encoder successfully converted to ONNX')


def convert_vae_decoder_onnx(vae: torch.nn.Module, onnx_path: Path, width:int = 64, height:int = 64):
    """
    Convert VAE model to ONNX, then IR format. 
    Function accepts pipeline, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        vae: 
        onnx_path (Path): File for storing onnx model
        width (int, optional, 64): input width
        height (int, optional, 64): input height
    Returns:
        None
    """
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            latents = 1 / 0.18215 * latents 
            return self.vae.decode(latents)

    if not onnx_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latents = torch.zeros((1, 4, width, height))

        vae_decoder.eval()
        with torch.no_grad():
            torch.onnx.export(vae_decoder, latents, onnx_path, input_names=[
                              'latents'], output_names=['sample'])
        print('VAE decoder successfully converted to ONNX')

# Helper code to convert models


def convert_txt_encoder_onnx_OV(model_dir, text_encoder):
    # Convert Text Encoder to ONNX then OpenVINO
    txt_encoder_onnx_path = model_dir / 'text_encoder.onnx'
    txt_encoder_ov_path = txt_encoder_onnx_path.with_suffix('.xml')

    if not txt_encoder_ov_path.exists():
        convert_encoder_onnx(text_encoder, txt_encoder_onnx_path)
        txt_encoder_ov_model = convert_model(txt_encoder_onnx_path)
        serialize(model=txt_encoder_ov_model, xml_path=str(txt_encoder_ov_path)) 
    else:
        print(f"Text encoder will be loaded from {txt_encoder_ov_path}")
    
    del text_encoder
    gc.collect();
    return txt_encoder_ov_path


def convert_unet_onnx_OV(model_dir, unet, num_channels=4, width=96, height=96):
    # Convert U-Net to ONNX then OpenVINO
    unet_onnx_path = model_dir / 'unet/unet.onnx'
    unet_ov_path = unet_onnx_path.parents[1] / 'unet.xml'

    if not unet_ov_path.exists():
        convert_unet_onnx(unet, unet_onnx_path, num_channels=num_channels, width=width, height=height)
        unet_ov_model = convert_model(unet_onnx_path)
        serialize(model=unet_ov_model, xml_path=str(unet_ov_path)) 
    else:
        print(f"U-Net will be loaded from {unet_ov_path}")
    
    del unet
    gc.collect();
    return unet_ov_path


def convert_vae_encoder_onnx_OV(model_dir, vae, width=768, height=768):
    # Converts the encoder VAE component to ONNX then OpenVINO
    vae_encoder_onnx_path = model_dir / 'vae_encoder.onnx'
    vae_encoder_ov_path = vae_encoder_onnx_path.with_suffix('.xml')

    if not vae_encoder_ov_path.exists():
        convert_vae_encoder_onnx(vae, vae_encoder_onnx_path, width=width, height=height)
        encoder_ov_model = convert_model(vae_encoder_onnx_path)
        serialize(model=encoder_ov_model, xml_path=str(vae_encoder_ov_path))
    else:
        print(f"VAE-Encoder will be loaded from {vae_encoder_ov_path}")   
    return vae_encoder_ov_path


def convert_vae_decoder_onnx_OV(model_dir, vae, width=96, height=96):
    # Converts the VAE decoder to ONNX then OpenVINO
    vae_decoder_onnx_path = model_dir / 'vae_decoder.onnx'
    vae_decoder_ov_path = vae_decoder_onnx_path.with_suffix('.xml')

    if not vae_decoder_ov_path.exists():
        convert_vae_decoder_onnx(vae, vae_decoder_onnx_path, width=width, height=height)
        decoder_ov_model = convert_model(vae_decoder_onnx_path)
        serialize(model=decoder_ov_model, xml_path=str(vae_decoder_ov_path)) 
    else:
        print(f"VAE decoder will be loaded from {vae_decoder_ov_path}")
    
    del vae
    gc.collect();
    return vae_decoder_ov_path
