# LLM-powered chatbot using Stable-Zephyr-3b and OpenVINO

In the rapidly evolving world of artificial intelligence (AI), chatbots have become powerful tools for businesses to enhance customer interactions and streamline operations. 
Large Language Models (LLMs) are artificial intelligence systems that can understand and generate human language. They use deep learning algorithms and massive amounts of data to learn the nuances of language and produce coherent and relevant responses.
While a decent intent-based chatbot can answer basic, one-touch inquiries like order management, FAQs, and policy questions, LLM chatbots can tackle more complex, multi-touch questions. LLM enables chatbots to provide support in a conversational manner, similar to how humans do, through contextual memory. Leveraging the capabilities of Language Models, chatbots are becoming increasingly intelligent, capable of understanding and responding to human language with remarkable accuracy.

`Stable Zephyr 3B` is a 3 billion parameter model that demonstrated outstanding results on many LLM evaluation benchmarks outperforming many popular models in relatively small size. Inspired by [HugginFaceH4's Zephyr 7B](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) training pipeline this model was trained on a mix of publicly available datasets, synthetic datasets using [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290), evaluation for this model based on [MT Bench](https://tatsu-lab.github.io/alpaca_eval/) and [Alpaca Benchmark](https://tatsu-lab.github.io/alpaca_eval/). More details about model can be found in [model card](https://huggingface.co/stabilityai/stablelm-zephyr-3b)

In this tutorial, we consider how to optimize and run this model using the OpenVINO toolkit. For the convenience of the conversion step and model performance evaluation, we will use [llm_bench](https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python) tool, which provides a unified approach to estimate performance for LLM. It is based on pipelines provided by [Optimum-Intel](https://github.com/huggingface/optimum-intel) and allows to estimate performance for Pytorch and OpenVINO models using almost the same code. We also demonstrate how to make model stateful using OpenVINO transformations, which improves process of caching model state.

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Download and convert the model
- Compress model weights to INT4 using [NNCF](https://github.com/openvinotoolkit/nncf)
- Estimate performance
- Estimate performance with applying stateful transformation
- Run interactive demo

The tutorial also provides interactive demo, where the model is used as a chatbot.

![stable_zephyr](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/cfac6ddb-6f22-4343-855c-e513269cf2bf)

## Installation Instructions
If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).