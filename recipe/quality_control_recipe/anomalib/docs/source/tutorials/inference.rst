.. _inference_documentation:

Inference
---------
Anomalib provides entrypoint scripts for using a trained model to generate predictions from a source of image data. This guide explains how to run inference with the standard PyTorch model and the exported OpenVINO model.


PyTorch (Lightning) Inference
=============================
The entrypoint script in ``tools/inference/lightning.py`` can be used to run inference with a trained PyTorch model. The script runs inference by loading a previously trained model into a PyTorch Lightning trainer and running the ``predict sequence``. The entrypoint script has several command line arguments that can be used to configure inference:

Like the other PyTorch Lightning entrypoints, the inference script reads the model configuration from a config file. It is generally advisible to use the same config file that was used when training the model, although some inference-related paremeter values may be changed to accommodate your needs. Some settings that you might want to change before inference are the batch size, transform config and number of workers for the dataloader.

It is also possible to run inference with a custom anomaly score threshold that differs from the threshold used/computed during training. To change the threshold for inference, simply set the ``metrics.threshold.method`` to ``manual`` and specify the custom threshold values in ``metrics.threshold.manual_image`` and ``metrics.threshold.manual_pixel``.

+---------------------+----------+---------------------------------------------------------------------------------+
|      Parameter      | Required |                                   Description                                   |
+=====================+==========+=================================================================================+
| config              | True     | Path to the model config file.                                                  |
+---------------------+----------+---------------------------------------------------------------------------------+
| weights             | True     | Path to the ``.ckpt`` model checkpoint file.                                    |
+---------------------+----------+---------------------------------------------------------------------------------+
| input               | True     | Path to the image source. This can be a single image or a folder of images.     |
+---------------------+----------+---------------------------------------------------------------------------------+
| output              | False    | Path to which the output images should be saved.                                |
+---------------------+----------+---------------------------------------------------------------------------------+
| visualization_mode  | False    | Determines how the inference results are visualized. Options: "full", "simple". |
+---------------------+----------+---------------------------------------------------------------------------------+
| show                | False    | When true, the visualized inference results will be displayed on the screen.    |
+---------------------+----------+---------------------------------------------------------------------------------+

To run inference, call the script from the command line with the with the following parameters, e.g.:

``python tools/inference/lightning.py --config padim.yaml --weights results/weights/model.ckpt --input image.png``

This will run inference on the specified image file or all images in the folder. A visualization of the inference results including the predicted heatmap and segmentation results (if applicable), will be displayed on the screen, like the example below.



OpenVINO Inference
==================
To run OpenVINO inference, first make sure that your model has been exported to the OpenVINO IR format. Once the model has been exported, OpenVINO inference can be triggered by running the OpenVINO entrypoint script in ``tools/inference/openvino.py``. The command line arguments are very similar to PyTorch inference entrypoint script:

+-----------+----------+--------------------------------------------------------------------------------------+
| Parameter | Required |                                     Description                                      |
+===========+==========+======================================================================================+
| config    | True     | Path to the model config file.                                                       |
+-----------+----------+--------------------------------------------------------------------------------------+
| weights   | True     | Path to the OpenVINO IR model file (either ``.xml`` or ``.bin``)                     |
+-----------+----------+--------------------------------------------------------------------------------------+
| image     | True     | Path to the image source. This can be a single image or a folder of images.          |
+-----------+----------+--------------------------------------------------------------------------------------+
| save_data | False    | Path to which the output images should be saved. Leave empty for live visualization. |
+-----------+----------+--------------------------------------------------------------------------------------+
| metadata | True     | Path to the JSON file containing the model's meta data (e.g. normalization           |
|           |          | parameters and anomaly score threshold).                                             |
+-----------+----------+--------------------------------------------------------------------------------------+
| device    | False    | Device on which OpenVINO will perform the computations (``CPU``, ``GPU`` or ``VPU``) |
+-----------+----------+--------------------------------------------------------------------------------------+

For correct inference results, the ``metadata`` argument should be specified and point to the ``metadata.json`` file that was generated when exporting the OpenVINO IR model. The file is stored in the same folder as the ``.xml`` and ``.bin`` files of the model.

As an example, OpenVINO inference can be triggered by the following command:

``python tools/inference/openvino.py --config padim.yaml --weights results/openvino/model.xml --input image.png --metadata results/openvino/metadata.json``

Similar to PyTorch inference, the visualization results will be displayed on the screen, and optionally saved to the file system location specified by the ``save_data`` parameter.



Gradio Inference
================

The gradio inference is supported for both PyTorch and OpenVINO models.

+-----------+----------+------------------------------------------------------------------+
| Parameter | Required |                           Description                            |
+===========+==========+==================================================================+
| config    | True     | Path to the model config file.                                   |
+-----------+----------+------------------------------------------------------------------+
| weights   | True     | Path to the OpenVINO IR model file (either ``.xml`` or ``.bin``) |
+-----------+----------+------------------------------------------------------------------+
| metadata | False    | Path to the JSON file containing the model's meta data.          |
|           |          | This is needed only for OpenVINO model.                          |
+-----------+----------+------------------------------------------------------------------+
| threshold | False    | Threshold value used for identifying anomalies. Range 1-100.     |
+-----------+----------+------------------------------------------------------------------+
| share     | False    | Share Gradio `share_url`                                         |
+-----------+----------+------------------------------------------------------------------+

To use gradio with OpenVINO model, first make sure that your model has been exported to the OpenVINO IR format and ensure that the `metadata` argument points to the ``metadata.json`` file that was generated when exporting the OpenVINO IR model. The file is stored in the same folder as the ``.xml`` and ``.bin`` files of the model.

As an example, PyTorch model can be used by the following command:

.. code-block:: bash

    python tools/inference/gradio_inference.py \
        --config ./anomalib/models/padim/config.yaml \
        --weights ./results/padim/mvtec/bottle/weights/model.ckpt

Similarly, you can use OpenVINO model by the following command:

.. code-block:: bash

    python python tools/inference/gradio_inference.py \
        --config ./anomalib/models/padim/config.yaml \
        --weights ./results/padim/mvtec/bottle/openvino/openvino_model.onnx \
        --metadata ./results/padim/mvtec/bottle/openvino/metadata.json
