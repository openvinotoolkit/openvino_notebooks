# Accelerate Inference of sparse NLP models with OpenVINO™ and 4th Gen Intel&reg; Xeon&reg; Scalable processors

This tutorial demonstrates how to improve performance of sparse NLP models with [OpenVINO](https://docs.openvino.ai/) on 4th Gen Intel® Xeon® Scalable processors. It uses a pre-trained model from the [HuggingFace Transformers](https://huggingface.co/transformers/) library and shows how to convert it to the OpenVINO™ IR format and run inference of the model on the CPU using a dedicated runtime option that enables sparsity optimizations. It also demonstrates how to get more performance stacking sparsity with 8-bit quantization. To simplify the user experience, the [HuggingFace Optimum](https://huggingface.co/docs/optimum) library is used to convert the model to the OpenVINO™ IR format and quantize it.

>**NOTE**: This tutorial requires OpenVINO 2022.3 or newer and 4th Gen Intel&reg; Xeon&reg; Scalable processor that can be acquired on Amazon AWS.

## Notebook Contents

The tutorial consists of the following steps:

* Download and convert the sparse BERT model.
* Compare sparse vs. dense inference performance.
* Quantize model.
* Compare sparse 8-bit vs. dense 8-bit inference performance.
