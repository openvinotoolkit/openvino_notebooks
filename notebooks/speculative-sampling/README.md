# Text Generation via Speculative Sampling, KV Caching, and OpenVINO™


[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/speculative-sampling/speculative-sampling.ipynb)

s model sizes grow, Generative AI implementations require significant inference resources. This not only increases the cost per generation from a prompt, but also increases the power consumption used to serve such requests.

Inference optimizations for text generation are essential for reducing costs and power consumption. When optimizing the inference process, the amount of time and energy required to generate text can be significantly reduced. This can lead to cost savings in terms of hardware and software, as well as reduced power consumption. Additionally, inference optimizations can help improve the accuracy of text generation as well as the speed at which it can be generated. This can lead to an improved user experience and increased efficiency in text-generation tasks. In summary, inference optimizations for text generation are essential to reduce costs and power consumption, while also improving the accuracy and speed of text generation.


Speculative decoding (or [assisted-generation](https://huggingface.co/blog/assisted-generation#understanding-text-generation-latency)) is a recent technique, that allows to speed up token generation when an additional smaller draft model is used alongside with the main model.

Speculative decoding works the following way. The draft model predicts the next K tokens one by one in an autoregressive manner, while the main model validates these predictions and corrects them if necessary. We go through each predicted token, and if a difference is detected between the draft and main model, we stop and keep the last token predicted by the main model. Then the draft model gets the latest main prediction and again tries to predict the next K tokens, repeating the cycle.

This approach reduces the need for multiple infer requests to the main model, enhancing performance. For instance, in more predictable parts of text generation, the draft model can, in best-case scenarios, generate the next K tokens that exactly match the target. In that case they are validated in a single inference request to the main model (which is bigger, more accurate but slower) instead of running K subsequent requests. More details can be found in the original [paper](https://arxiv.org/pdf/2211.17192.pdf).

![](https://github.com/user-attachments/assets/eb999dea-d98b-42bb-835e-28d3054e1a84)

In this tutorial we consider how to apply Speculative decoding using OpenVINO GenAI.

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Download models
- Run speculative sampling example and compare speed-up with respect to autoregressive sampling.

## Installation instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).


## Acknowledgement

A numpy version of speculative sampling is available from Mody at https://jaykmody.com/blog/speculative-sampling/ - while our code was written from scratch, we did make use of this code as a validation point for the technique.

## References
[1] Pope et al, *Efficiently Scaling Transformer Inference,* http://arxiv.org/abs/2211.05102

[2] Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need,* http://arxiv.org/abs/1911.02150

[3] Schuster et al, *Confident Adaptive Language Modeling,* https://arxiv.org/abs/2207.07061

[4] Belrose et al, *Eliciting Latent Predictions from Transformers with the Tuned Lens,* http://arxiv.org/abs/2303.08112

[5] Chen et al, *Accelerating Large Language Model Decoding with Speculative Sampling,* http://arxiv.org/abs/2302.01318

[6] Kim et al, *Big Little Transformer Decoder,*  http://arxiv.org/abs/2302.07863

[7] Gante, Joao, *Assisted Generation: a new direction toward low-latency text generation,* https://huggingface.co/blog/assisted-generation

[8] Stern et al, *Blockwise Parallel Decoding for Deep Autoregressive Models,* http://arxiv.org/abs/1811.03115

[9] Lai et al, *Understanding Autoregressive Model for Time Series as a Deterministic Dynamic System,*  https://www.soa.org/globalassets/assets/library/newsletters/predictive-analytics-and-futurism/2017/june/2017-predictive-analytics-iss15-lai-lu.pdf


[def]: SpeculativeSampling.png
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/speculative-sampling/README.md" />
