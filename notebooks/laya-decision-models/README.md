# Laya Decision Models with OpenVINO™

[Laya](https://huggingface.co/convaiinnovations/laya) is a multilingual, non-autoregressive **System 1 decision model** from [Convai Innovations](https://huggingface.co/convaiinnovations). Instead of generating text token by token, it scores answer options in a single forward pass, which makes it a fast and reliable building block for routing, triage and filtering tasks:

* **Single-pass decisions**: no autoregressive decoding — one forward pass scores all answer options at once.
* **No hallucinations by design**: the model never generates free-form text, it only scores the options you provide.
* **Three question types**: `choice` (pick one of N options), `score` (rate on a fixed scale) and `noul` (none-of-the-above / binary uncertainty).
* **Calibrated probabilities**: trained and calibrated with RLCD (RL from Contrastive Distillation), so confidence scores are meaningful.
* **Multilingual**: strong out-of-the-box quality in English plus 100+ other languages.

More details can be found in the original [model card](https://huggingface.co/convaiinnovations/laya).

In this tutorial, we will:
1. Install required dependencies
2. Download the Laya checkpoint from the Hugging Face Hub
3. Convert the model to OpenVINO IR with `optimum-cli`, choosing between fp16, int8 and int4 weight compression
4. Run an email-triage example with the official `system_one` API on CPU or Intel GPU, measure inference latency and triage custom emails interactively

The notebook installs `optimum-intel` from the [fork branch](https://github.com/xiangfuxwang/optimum-intel-dev/tree/laya) that adds Laya export and inference support, until it is upstreamed.
