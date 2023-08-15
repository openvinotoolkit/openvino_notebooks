# Text-to-video generation with ZeroScope and OpenVINO

The ZeroScope model is a free and open-source text-to-video model that can generate realistic and engaging videos from text descriptions. It is based on the Modelscope model, but it has been improved to produce higher-quality videos with a 16:9 aspect ratio and no Shutterstock watermark. The ZeroScope model is available in two versions: Zeroscope_v2 576w, which is optimized for rapid content creation at a resolution of 576x320 pixels, and Zeroscope_v2 XL, which upscales videos to a high-definition resolution of 1024x576.

The ZeroScope model is trained on a dataset of over 9,000 videos and 29,000 tagged frames. It uses a diffusion model to generate videos, which means that it starts with a random noise image and gradually adds detail to it until it matches the text description. The ZeroScope model is still under development, but it has already been used to create some impressive videos. For example, it has been used to create videos of people dancing, playing sports, and even driving cars.

The ZeroScope model is a powerful tool that can be used to create various videos, from simple animations to complex scenes. It is still under development, but it has the potential to revolutionize the way we create and consume video cont

Both versions of the ZeroScope model are available on Hugging Face:

- Zeroscope_v2 576w
- Zeroscope_v2 XL We will use the first one.
Table of content:

- 1. Install and import required packages
- 2. Load the model
- 3. Convert the model
    - 3.1. Define the conversion function
    - 3.2. UNet
    - 3.3. VAE
    - 3.4. Text encoder
- 4. Build a pipeline
- 5. Inference with OpenVINO
    - 5.1. Select inference device
    - 5.2. Define a prompt
    - 5.3. Video generation