from pathlib import Path
import gc
import torch
import numpy as np
import openvino as ov


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def convert_encoder(text_encoder: torch.nn.Module, ir_path:Path):
    """
    Convert Text Encoder model to IR. 
    Function accepts pipeline, prepares example inputs for conversion 
    Parameters: 
        text_encoder (torch.nn.Module): text encoder PyTorch model
        ir_path (Path): File for storing model
    Returns:
        None
    """
    if not ir_path.exists():
        input_ids = torch.ones((1, 77), dtype=torch.long)
        # switch model to inference mode
        text_encoder.eval()

        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            # export model
            ov_model = ov.convert_model(
                text_encoder,  # model instance
                example_input=input_ids,  # example inputs for model tracing
                input=([1,77],)  # input shape for conversion
            )
            ov.save_model(ov_model, ir_path)
            del ov_model
            cleanup_torchscript_cache()
        print('Text Encoder successfully converted to IR')

        
def convert_unet(unet:torch.nn.Module, ir_path:Path, num_channels:int = 4, width:int = 64, height:int = 64):
    """
    Convert Unet model to IR format. 
    Function accepts pipeline, prepares example inputs for conversion 
    Parameters: 
        unet (torch.nn.Module): UNet PyTorch model
        ir_path (Path): File for storing model
        num_channels (int, optional, 4): number of input channels
        width (int, optional, 64): input width
        height (int, optional, 64): input height
    Returns:
        None
    """
    dtype_mapping = {
        torch.float32: ov.Type.f32,
        torch.float64: ov.Type.f64
    }
    if not ir_path.exists():
        # prepare inputs
        encoder_hidden_state = torch.ones((2, 77, 1024))
        latents_shape = (2, num_channels, width, height)
        latents = torch.randn(latents_shape)
        t = torch.from_numpy(np.array(1, dtype=np.float32))
        unet.eval()
        dummy_inputs = (latents, t, encoder_hidden_state)
        input_info = []
        for input_tensor in dummy_inputs:
            shape = ov.PartialShape(tuple(input_tensor.shape))
            element_type = dtype_mapping[input_tensor.dtype]
            input_info.append((shape, element_type))

        with torch.no_grad():
            ov_model = ov.convert_model(
                unet, 
                example_input=dummy_inputs,
                input=input_info
            )
        ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        print('U-Net successfully converted to IR')


def convert_vae_encoder(vae: torch.nn.Module, ir_path: Path, width:int = 512, height:int = 512):
    """
    Convert VAE model to IR format. 
    VAE model, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for onversion 
    Parameters: 
        vae (torch.nn.Module): VAE PyTorch model
        ir_path (Path): File for storing model
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
            return self.vae.encode(x=image)["latent_dist"].sample()

    if not ir_path.exists():
        vae_encoder = VAEEncoderWrapper(vae)
        vae_encoder.eval()
        image = torch.zeros((1, 3, width, height))
        with torch.no_grad():
            ov_model = ov.convert_model(vae_encoder, example_input=image, input=([1,3, width, height],))
        ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        print('VAE encoder successfully converted to IR')


def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path, width:int = 64, height:int = 64):
    """
    Convert VAE decoder model to IR format. 
    Function accepts VAE model, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for conversion 
    Parameters: 
        vae (torch.nn.Module): VAE model 
        ir_path (Path): File for storing model
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
            return self.vae.decode(latents)

    if not ir_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latents = torch.zeros((1, 4, width, height))

        vae_decoder.eval()
        with torch.no_grad():
            ov_model = ov.convert_model(vae_decoder, example_input=latents, input=([1,4, width, height],))
        ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        print('VAE decoder successfully converted to IR')
