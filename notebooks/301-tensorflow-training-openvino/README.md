# Training to Deployment with TensorFlow and OpenVINO™

|  |  |
|---|---|
| ![f](https://www.tensorflow.org/tutorials/images/classification_files/output_N1loMlbYHeiJ_0.png) | ![f](https://www.tensorflow.org/tutorials/images/classification_files/output_RQbZBOTLHiUP_0.png) |
| ![f](https://www.gstatic.com/knowyourdata/20210519-a7914a/tf_flowers/media/dHJhaW5bMCU6MiVdXzE3.jpeg) | ![f](https://www.tensorflow.org/tutorials/images/classification_files/output_HyQkfPGdHilw_0.png) |

In this directory, you will find two Jupyter notebooks. The first is an end-to-end deep learning training tutorial which borrows the open source code from the TensorFlow [image classification tutorial](https://www.tensorflow.org/tutorials/images/classification), demonstrating how to train the model and then convert to OpenVINO Intermediate Representation (OpenVINO IR). It leverages the [tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) dataset which includes about 3,700 photos of flowers.

The second notebook demonstrates how to quantize the OpenVINO IR model that was created in the first notebook. Post-training quantization speeds up inference on the trained model. The quantization is performed with the [Post Training Optimization Tool (POT)](https://docs.openvino.ai/latest/pot_introduction.html) from OpenVINO Toolkit. A custom dataloader and a metric will be defined, and accuracy and performance will be computed for the original OpenVINO IR model and the quantized model on CPU and iGPU (if available).

## Jupyter Notebooks

* `301-tensorflow-training-openvino.ipynb`
  * Demonstrates how to train, convert, and deploy an image classification model with TensorFlow and OpenVINO.
* `301-tensorflow-training-openvino-pot.ipynb`
  * Demonstrates how to quantize the OpenVINO IR model that was created by the previous notebook.

## TensorFlow Licenses

These notebooks are based on the [TensorFlow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification), distributed with an Apache 2.0 License, displayed below. The images in the TensorFlow Flowers dataset are Licensed with [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). See the `LICENSE.txt` file inside the `flower_photos.tgz` archive.

You may not use this file except in compliance with the License.
You may obtain a copy of the [License at apache.org](https://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

```license
@ONLINE {tfflowers,
author = "The TensorFlow Team",
title = "Flowers",
month = "jan",
year = "2019",
url = "http://download.tensorflow.org/example_images/flower_photos.tgz" }
```
