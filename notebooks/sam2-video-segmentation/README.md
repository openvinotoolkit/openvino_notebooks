# Video segmentation with SAM2 and OpenVINO™


Segmentation - identifying which image pixels belong to an object - is a core task in computer vision and is used in a broad array of applications, from analyzing scientific imagery to editing photos. But creating an accurate segmentation model for specific tasks typically requires highly specialized work by technical experts with access to AI training infrastructure and large volumes of carefully annotated in-domain data. Reducing the need for task-specific modeling expertise, training compute, and custom data annotation for image segmentation is the main goal of the Segment Anything project.

[Segment Anything Model 2 (SAM 2)](https://ai.meta.com/sam2/) is a foundation model towards solving promptable visual segmentation in images and videos. It extend SAM to video by considering images as a video with a single frame. The SAM 2 model extends the promptable capability of SAM to the video domain by adding a per session memory module that captures information about the target object in the video. This allows SAM 2 to track the selected object throughout all video frames, even if the object temporarily disappears from view, as the model has context of the object from previous frames. SAM 2 also supports the ability to make corrections in the mask prediction based on additional prompts on any frame. SAM 2’s streaming architecture—which processes video frames one at a time—is also a natural generalization of SAM to the video domain. When SAM 2 is applied to images, the memory module is empty and the model behaves like SAM. 

The model design is a simple transformer architecture with streaming memory for real-time video processing. The model is built a model-in-the-loop data engine, which improves model and data via user interaction, to collect SA-V dataset, the largest video segmentation dataset to date. SAM 2 provides strong performance across a wide range of tasks and visual domains.
This notebook shows an example of how to convert and use Segment Anything Model 2 in OpenVINO format, allowing it to run on a variety of platforms that support an OpenVINO.

* Interactive segmentation mode: in this demonstration you can upload video and specify point/box related to object using [Gradio](https://gradio.app/) interface and as the result you get segmentation mask for specified point.
The following image shows an example of the input image and the corresponding predicted image.
![demo](https://user-images.githubusercontent.com/29454499/231464914-bd2a683c-28b2-44d4-960e-dce3e3ddebc3.png)


## Notebook Contents

This notebook shows an example of how to convert and use Segment Anything Model using OpenVINO

Notebook contains the following steps:
1. Convert PyTorch models to OpenVINO format.
2. Run OpenVINO model in interactive segmentation mode.


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/sam2-video-segmentation/README.md" />