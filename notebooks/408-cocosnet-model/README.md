# AI photo-realistic Synthesis with OpenVINO™

This notebook demonstrates an exemplar-based image translation that synthesizes a photo-realistic image based on an exemplar image. The users provide a reference image and draw a sketch using an interactive canvas to get a realistic photo based on the provided semantic drawings.

## Notebook Contents

This notebook uses [CoCosNet](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/cocosnet) model from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master) for exemplar-based image translation which synthesizes a photo-realistic image from the input and [Gradio](https://gradio.app/docs/) for creating a web-based GUI to synthesize an image based on the drawing input on the Gradio canvas and the uploaded reference image. The notebook shows how to create the following pipeline:

[Ai-painting-pipeline](data/Ai-painting.png)

Notebook contains the following steps:

1. Download the Model
2. Convert Model to OpenVINO IR format
3. Model Initialization
4. Input preprocessing and Model inferencing
5. Gradio interface

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README.md).

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)