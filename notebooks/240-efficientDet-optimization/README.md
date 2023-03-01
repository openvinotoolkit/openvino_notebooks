## EfficientDet Optimization with OpenVINO
<a href='https://github.com/google/automl/tree/master/efficientdet'>EfficientDets</a> are a family of object detection models, which achieve state-of-the-art 55.1mAP on COCO test-dev, yet being 4x - 9x smaller and using 13x - 42x fewer FLOPs than previous detectors. Our models also run 2x - 4x faster on GPU, and 5x - 11x faster on CPU than other detectors

### The tutorial consists of the following steps:

* Downloading and exporting efficientDet-d0 model
* loading efficintDet-d0 into tensorflow
  * Validating inference of tf-model
* Converting tf-model to OpenVINO-IR(Intermidiate representation) file {FP16}
  * Validating inference of OpenVINO-model
  * Evaluating performance by AP( average Percesion)
* Post-training optimization with OpenVINO NNCF (Neural Network Compression Framework)
  * Quantization of FP16 model to INT8 model
  * Validating Inference of INT8 model
  * Evaluating performance of INT8 model by AP
* Compare performance of the FP32 and quantized model with OpenVINO benchmark_app

<img src='result_img/ir_inference.png'>

#### Optimization Result
* FP16 model : 4.64 FPS or images processed per second
* INT8 model : 6.28 FPS
