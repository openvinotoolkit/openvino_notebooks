# README

## Initial Setup: Model Download Steps

To run the notebook and demo, you will need to have the following models pre-downloaded.
You can run the scripts in the TBD.ipynb file to download and convert these models to IR.

- Whisper:
	- whisper_models/whisper_encoder.xml
	- whisper_models/whisper_encoder.bin
	- whisper_models/whisper_decoder.xml
	- whisper_models/whisper_decoder.bin
- CLIP for visual language saliency:
	- clip-vit-base-patch16_text.xml
	- clip-vit-base-patch16_text.bin
	- clip-vit-base-patch16_image.xml
	- clip-vit-base-patch16_image.bin

The model files for Stable Diffusion XL and RedPajama-INCITE will be downloaded via the main demo notebook.

The utils/ folder contains the scripts needed for pre/post-processing and inference of the OpenVINO models for the notebook and Gradio demo.