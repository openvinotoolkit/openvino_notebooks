# Image generation with Sana and OpenVINO

**Sana** is a text-to-image framework that can efficiently generate images up to 4096 × 4096 resolution developed by NVLabs. Sana can synthesize high-resolution, high-quality images with strong text-image alignment at a remarkably fast speed, deployable on laptop GPU. 
Core designs include: 
* **Deep compression autoencoder**: unlike traditional AEs, which compress images only 8×, we trained an AE that can compress images 32×, effectively reducing the number of latent tokens.
* **Linear DiT**: authors replaced all vanilla attention in DiT with linear attention, which is more efficient at high resolutions without sacrificing quality.
* **Decoder-only text encoder***: T5 replaced by modern decoder-only small LLM as the text encoder and designed complex human instruction with in-context learning to enhance the image-text alignment.
* **Efficient training and sampling**: Proposed Flow-DPM-Solver to reduce sampling steps, with efficient caption labeling and selection to accelerate convergence.

More details about model can be found in [paper](https://arxiv.org/abs/2410.10629), [model page](https://nvlabs.github.io/Sana/) and [original repo](https://github.com/NVlabs/Sana).

**SANA-1.5** is a linear Diffusion Transformer for efficient scaling in text-to-image generation. SANA-1.5 is built on SANA-1.0 with introduction following improvements:
* **Efficient Training Scaling**: A depth-growth paradigm that enables scaling from 1.6B to 4.8B parameters with significantly reduced computational resources, combined with a memory-efficient 8-bit optimizer.
* **Model Depth Pruning**: A block importance analysis technique for efficient model compression to arbitrary sizes with minimal quality loss.
* **Inference-time Scaling**: A repeated sampling strategy that trades computation for model capacity, enabling smaller models to match larger model quality at inference time.

More details about model can be found in [paper](https://arxiv.org/abs/2501.18427), [model page](https://nvlabs.github.io/Sana/Sana-1.5/) and [original repo](https://github.com/NVlabs/Sana).

**SANA-Sprint** is an efficient diffusion model for ultra-fast text-to-image.  SANA-Sprint is built on a pre-trained foundation model and augmented with hybrid distillation, dramatically reducing inference steps from 20 to 1-4. 
Core innovations include:
* **Training-Free Transformation to TrigFlow**: the paper proposes a training-free approach that transforms a pre-trained flow-matching model for continuous-time consistency distillation (sCM), eliminating costly training from scratch and achieving high training efficiency. The hybrid distillation strategy combines sCM with latent adversarial distillation (LADD): sCM ensures alignment with the teacher model, while LADD enhances single-step generation fidelity.
* **Stabilizing Continuous-Time Distillation**: To stabilize continuous-time consistency distillation, we address two key challenges: training instabilities and excessively large gradient norms that occur when scaling up the model size and increasing resolution, leading to model collapse. We achieve this by refining dense time-embedding and integrating QK-Normalization into self- and cross-attention mechanisms. These modifications enable efficient training and improve stability, allowing for robust performance at higher resolutions and larger model sizes. 
* **Improving Continuous-Time CMs with GAN**: CTM analyzes that CMs distill teacher information in a local manner, where at each iteration, the student model learns from local time intervals. This leads the model to learn cross timestep information under the implicit extrapolation, which can slow the convergence speed. To address this limitation, we introduce an additional adversarial loss to provide direct global supervision across different timesteps, improving both the convergence speed and the output quality.

More details about model can be found in [paper](https://arxiv.org/pdf/2503.09641), [model page](https://nvlabs.github.io/Sana/Sprint/) and [original repo](https://github.com/NVlabs/Sana).

In this tutorial, we consider how to optimize and run models from Sana's family using OpenVINO.

### Notebook Contents

In this demonstration, you will learn how to perform text-to-image generation using Sana and OpenVINO. 

Example of model work:

**Input prompt**: *a cyberpunk cat with a neon sign that says "Sana"*
![](https://github.com/user-attachments/assets/bacfcd2a-ac36-4421-9d1b-4e34aa0a9f62)

The tutorial consists of the following steps:

- Install prerequisites
- Collect Pytorch model pipeline
- Convert model to OpenVINO intermediate representation (IR) format
- Compress model weights using NNCF
- Prepare OpenVINO Inference pipeline
- Run Text-to-Image generation
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/sana-image-generation/README.md" />
