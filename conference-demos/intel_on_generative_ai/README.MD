# README

![Hybrid AI Demo Flow](https://github.com/QData/TextAttack/assets/22090501/eb9a2e7e-504c-4e90-aeca-b3147ab3b0c5)
    
    In this demo, we'll chain multiple Generative AI models together in OpenVINO, with runtime in seconds on Intel CPUs and GPUs! The pipeline is composed of four models: 
    
    1. Whisper for speech transcription
    2. RedPajama-INCITE (chat version - 3B parameters) for refinement of the generated text
    3. Stable Diffusion (options: v2.1 and XL) for using the text as a prompt for image generation
    4. CLIP to explore interpretability of the generated image. 


## PLEASE READ - Initial Setup: Model Download Steps


Running the notebook: To run the notebook, **you will need to have the model files for Whisper and CLIP pre-downloaded.** The model files for Stable Diffusion XL and RedPajama-INCITE will be downloaded via the main demo notebook.

Running the Gradio app: Please run the notebook prior to running the Gradio app to download the required models for the demo.

The utils/ folder contains the scripts needed for pre/post-processing and inference of the OpenVINO models for the notebook and Gradio demo.