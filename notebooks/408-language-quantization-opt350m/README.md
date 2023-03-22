# Quantize OPT : Open Pre-trained Transformer Language Models with Post-Training Optimization Tool ​in OpenVINO™
This tutorial demonstrates how to apply `INT8` quantization to the [OPT350m](https://huggingface.co/facebook/opt-350m), using the [Post-Training Optimization Tool API](https://docs.openvino.ai/latest/pot_compression_api_README.html) (part of the [OpenVINO Toolkit](https://docs.openvino.ai/)). [Microsoft Research Paraphrase Corpus (MRPC)](https://www.microsoft.com/en-us/download/details.aspx?id=52398) dataset is used for quantization.
Structure of the notebook is as follows:

- Download and prepare the OPT350m model and MRPC dataset.
- Define data loading and accuracy validation functionality. Accuracy checking was not performed due to memory limits
- Prepare the model for quantization.
- Run optimization pipeline.
- Load and test quantized model.
- Compare the performance of the original, converted and quantized models.

# Speedup
 - PyTorch model on CPU: 1.240 seconds per sentence
 - IR FP32 model in OpenVINO Runtime/CPU: 0.443 seconds per sentence
 - OpenVINO IR INT8 model in OpenVINO Runtime/CPU: 0.165 seconds per sentence

## Using benchmark_app with default parameters
### Simple OpenVino IR model:
 - [ INFO ]    Average:       438.01 ms
 - [ INFO ]    Min:           429.08 ms
 - [ INFO ]    Max:           480.90 ms
 - [ INFO ] Throughput:   2.31 FPS

### INT8 quantized OpenVino IR model:
 - [ INFO ]    Average:       160.42 ms
 - [ INFO ]    Min:           155.11 ms
 - [ INFO ]    Max:           183.11 ms
 - [ INFO ] Throughput:   6.32 FPS