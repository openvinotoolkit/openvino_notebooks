# Instruction following using Databricks Dolly 2.0 and OpenVINO™

The instruction following is one of the cornerstones of the current generation of large language models(LLMs). Reinforcement learning with human preferences ([RLHF](https://arxiv.org/abs/1909.08593)) and techniques such as [InstructGPT](https://arxiv.org/abs/2203.02155) have been the core foundation of breakthroughs such as ChatGPT and GPT-4. However, these powerful models remain hidden behind APIs and we know very little about its underlying architecture. Instruction-following models are capable of generating text in response to prompts and are often used for tasks like writing assistance, chatbots, and content generation. Many users now interact with these models regularly and even use them for work but majority of them remains closed-source and requires massive amounts of computational resources to experiment with.

[Dolly 2.0](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) is the first open-source, instruction-following LLM fine-tuned by Databricks on a transparent and freely available dataset that is also open-sourced to use for commercial purposes. That means Dolly 2.0 is available for commercial applications without the need to pay for API access or share data with third parties. Dolly 2.0 exhibits similar characteristics so ChatGPT despite being much smaller.

In this tutorial we consider how to run instruction-following text generation pipeline using Dolly 2.0 and OpenVINO. We will use pre-trained model from the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library. To simplify the user experience, the [Hugging Face Optimum](https://huggingface.co/docs/optimum) library is used to convert the models to OpenVINO™ IR format.

The notebook provides simple interface which allowing communicate with model using text instruction. In this demonstration user can provide input instruction and model generates answer in streamin format.

The image below illustrates provided user instruction and model answer example.

![example](https://user-images.githubusercontent.com/29454499/237175794-51e5d9d2-6104-4cf9-972e-56c99e177388.png)


## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Download and convert model from a public source using the [OpenVINO integration with Hugging Face Optimum](https://huggingface.co/blog/openvino).
- Create an instruction-follwoing inference pipeline
- Run instruction-following pipeline


If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).