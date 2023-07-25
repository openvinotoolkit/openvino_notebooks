# Create LLM-powered Chatbot using OpenVINO

In the rapidly evolving world of artificial intelligence (AI), chatbots have emerged as powerful tools for businesses to enhance customer interactions and streamline operations. 
Large Language Models (LLMs) are artificial intelligence systems that can understand and generate human language. They use deep learning algorithms and massive amounts of data to learn the nuances of language and produce coherent and relevant responses.
While a decent intent-based chatbot can answer basic, one-touch inquiries like order management, FAQs, and policy questions, LLM chatbots can tackle more complex, multi-touch questions. LLM enables chatbots to provide support in a conversational manner, similar to how humans do, through contextual memory. Leveraging the capabilities of Language Models, chatbots are becoming increasingly intelligent, capable of understanding and responding to human language with remarkable accuracy.

Previously, we already discussed how to build instruction-following pipeline using OpenVINO and Optimum Intel, please check out [Dolly v2 example](../240-dolly-2-instruction-following) for reference.
In this tutorial we consider how to use power of OpenVINO for running Large Language Models for chat. We will use a pre-trained model from the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library. To simplify the user experience, the [Hugging Face Optimum Intel](https://huggingface.co/docs/optimum/intel/index) library is used to convert the models to OpenVINO™ IR format.

The tutorial supports different models, you can select one from provided options to compare quality of open source LLM solutions.
>**Note**: conversion of some models requries can requires additional actions from user side and at least 64GB RAM for conversion.

The available options are:

* **red-pijama-3b-chat** - A 2.8B parameter pretrained language model based on GPT-NEOX architecture. It was developed by Together Computer and leaders from the open-source AI community. The model is fine-tuned on OASST1 and Dolly2 datasets to enhance chatting ability. More details about model can be found in [HuggingFace model card](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1).
* **llama-2-7b-chat** - LLama 2 is the second generation of LLama models developed by Meta. Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. llama-2-7b-chat is 7 billions parameters version of LLama 2 finetuned and optimized for dialogue use case. More details about model can be found in the [paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/), [repository](https://github.com/facebookresearch/llama) and [HuggingFace model card](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
>**Note**: run model with demo, you will need to accept license agreement. 
>You must be a registered user in 🤗 Hugging Face Hub. Please visit [HuggingFace model card](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), carefully read terms of usage and click accept button.  You will need to use an access token for downloading model. For more information on access tokens, refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).
* **mpt-7b-chat** - MPT-7B is part of the family of MosaicPretrainedTransformer (MPT) models, which use a modified transformer architecture optimized for efficient training and inference. These architectural changes include performance-optimized layer implementations and the elimination of context length limits by replacing positional embeddings with Attention with Linear Biases ([ALiBi](https://arxiv.org/abs/2108.12409)). Thanks to these modifications, MPT models can be trained with high throughput efficiency and stable convergence. MPT-7B-chat is a chatbot-like model for dialogue generation. It was built by finetuning MPT-7B on the [ShareGPT-Vicuna](https://huggingface.co/datasets/jeffwan/sharegpt_vicuna), [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3), [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf), and [Evol-Instruct](https://huggingface.co/datasets/victor123/evol_instruct_70k) datasets. More details about model can be found in [blogpost](https://www.mosaicml.com/blog/mpt-7b), [repository](https://github.com/mosaicml/llm-foundry/) and [HuggingFace model card](https://huggingface.co/mosaicml/mpt-7b-chat).

The image below illustrates provided user instruction and model answer examples.

![example](https://user-images.githubusercontent.com/29454499/237291423-022f07d2-966b-4be2-9a1c-98f1cf0691c2.png)


## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Download and convert the model from a public source using the [OpenVINO integration with Hugging Face Optimum](https://huggingface.co/blog/openvino).
- Create an inference pipeline
- Run chatbot




If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).