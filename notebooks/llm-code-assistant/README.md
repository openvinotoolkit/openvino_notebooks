# LLM Code Assistant with OpenVINO™

This notebook demonstrates how to use **code-specialized Large Language Models** for practical coding tasks, running locally on Intel hardware with [OpenVINO™](https://github.com/openvinotoolkit/openvino).

## Notebook Contents

We use code-specialized models from the [Qwen2.5-Coder](https://huggingface.co/collections/Qwen/qwen25-coder) (7B–14B) and [Qwen3-Coder](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) families with the high-performance [`openvino-genai`](https://github.com/openvinotoolkit/openvino.genai) inference library. Pre-converted OpenVINO models are downloaded from the [OpenVINO collection on HuggingFace](https://huggingface.co/OpenVINO) when available; otherwise, models are converted locally using [Optimum Intel](https://huggingface.co/docs/optimum/intel/index).

**Demonstrations include:**
- **Bug Detection & Code Correction** — fix multiple bugs in Python functions with automatic test verification.
- **Interactive Web App Generation** — generate a functional Snake game from a short prompt, playable in the notebook.
- **Security Audit Agent** — agentic workflow to find vulnerabilities (SQL injection, hardcoded credentials, path traversal, command injection), fix them, and self-correct via AST verification.
- **Codebase Explorer Agent** — agentic workflow where the model autonomously explores a GitHub repository, decides which files to fetch, and builds a comprehensive analysis.
- **Interactive Chat** — pair-program with the code assistant via a Gradio interface.

## Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/llm-code-assistant/README.md" />
