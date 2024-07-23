# Text Generation via Speculative Sampling, KV Caching, and OpenVINO™


[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/speculative-sampling/speculative-sampling.ipynb)

As model sizes grow, Generative AI implementations require significant inference resources. This not only increases the cost per generation from a prompt, but also increases the power consumption used to serve such requests.

Inference optimizations for text generation are essential for reducing costs and power consumption. When optimizing the inference process, the amount of time and energy required to generate text can be significantly reduced. This can lead to cost savings in terms of hardware and software, as well as reduced power consumption. Additionally, inference optimizations can help improve the accuracy of text generation as well as the speed at which it can be generated. This can lead to an improved user experience and increased efficiency in text-generation tasks. In summary, inference optimizations for text generation are essential to reduce costs and power consumption, while also improving the accuracy and speed of text generation.

Another necessary condition is that the optimizations are compatible with each other. That is, implementing a certain optimization should not preclude other optimizations. There are several levels of optimizations that can provide significant speedup without "bumping into each other" in a way that will compromise overall efficiency.

## What is the problem?

Often multiple optimizations are possible to gain maximum advantage. Let us take a look at one such collection of optimizations that would be desired for many text generation applications. We will consider the following optimizations given a full model:

- **Model-Based Optimizations** - often we don't expect problems to arise if quantization (a commonly used model-based optimization) was tested by itself. However, there could be some unexpected effects from the quantization and sampling of models. A full discussion on quantization is beyond the scope of this article (but is assumed to be essential in high-performance inference); we will focus on dynamic execution methods in this work.
- **KV Caching** (or Past-Value Caching) - autoregressive sampling predicts the next value in a series of tokens. Computations are performed on these tokens to create the prediction of the next token. The expanded collection of tokens (all previously appended with the newly generated token from the previous pass) now goes through another pass, and this continues until the number of tokens requested is reached. To avoid a lot of repetitive calculations on the past tokens, the intermediate values are stored in a KV cache [1]. This method is very standard (enabled in HuggingFace by default) and poses no risk to accuracy. The only downside is that the KV cache can be quite large and increase memory requirements for the autoregressive process.
- **Speculative Sampling** - A form of dynamic execution, there has been a lot of published research in this area about using a smaller, draft model to produce samples that should be "good enough" much of the time, and occasionally reject candidates and pay the price of the full model when needed. This method has been published, with slight differences, in several independent research publications. [3] [4] [5] [6] [7] [8] 

In order to gain an appreciation of why speculative sampling works, let us take a step back and visit Autoregressive Sampling, combined with KV caching. We will see that the process is memory-bound allowing us to essentially test K tokens on the target model, in parallel for the same cost as sampling just one token. So having a decent acceptance rate means that many of the tokens are generated fast enough to compensate for the extra overhead of generating on a draft model and then checking in a target model.

## Autoregressive vs Speculative Sampling
A popular method of text generation is to generate next tokens based upon a probability conditioned on previous tokens, as given by:

$$p(\tilde{x}_{n+1} | x_1, ..., x_n)$$

This is known as autoregressive sampling [9] and is now a standard method of text-generation in generative models. This could be followed by one of several methods to select the token at $n+1$, for example, argmax or randomly selected from top-p. 

<p align="center"><img alt="Speculative Sampling" src="https://user-images.githubusercontent.com/29454499/280659301-49a38beb-e6f3-4a2c-858e-be4ca4491016.png" /> 
<br />Figure 1: Speculative Sampling Flow</p><br />

Note that sampling of models is memory intensive. Shazeer [2] shows that the ratio of memory access to arithmetic operations is very memory intensive for transformer-based sequential sampling. Chen et al. [5]] attribute the overall sampling time for large transformer-based models to linear layers, attention, and collective operations (all-reduce). We focus on a batch size of one for inference, but we can leverage a batch size of K words (sampled from a smaller draft model) to be evaluated in the target model together, taking about the same time as sampling a single token from the target model. For a reasonable value of K, we can, therefore, leverage the smaller draft model for much of the text generation, using the target model less often for evaluation (i.e., acceptance or rejection) and single token generation when rejection occurs. We have seen a significant increase in throughput using this method.

However, the draft model and target model have different sizes that would be represented in a KV cache, so the challenge is to take advantage of separate optimization strategies simultaneously. For this article, we assume a quantized model and leverage KV caching together with Speculative Sampling.

Note that the authors [5] prove that the target distribution is recovered when performing speculative sampling - this guarantees the same sampling quality as autoregressive sampling on the target itself. Therefore, the situations for not leveraging speculative sampling is not worthwhile have to do with the case where there are not enough savings in the relative size of the draft model or the acceptance rate of the draft model is not high enough to benefit from the smaller size of the draft model.

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Download and convert the model from a public source using the [OpenVINO integration with Hugging Face Optimum](https://huggingface.co/blog/openvino).
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