# ModernBERT Masked Language Modeling with OpenVINO

This notebook demonstrates how to convert and run the [ModernBERT](https://huggingface.co/blog/modernbert) model using OpenVINO. ModernBERT is an encoder-only model optimized for long-context understanding (up to 8k tokens) and efficiency.

## Notebook Contents

The notebook covers:
1.  Loading **ModernBERT** from Hugging Face.
2.  Converting the PyTorch model to **OpenVINO IR** format.
3.  Running inference on CPUs and other Intel hardware.
4.  Launch an interactive **Gradio** demo for masked language modeling.

## Installation Instructions

1.  Clone the repository:
    ```bash
    git clone https://github.com/openvinotoolkit/openvino_notebooks.git
    cd openvino_notebooks
    ```

2.  Install dependencies:
    ```bash
    pip install -r .ci/dev-requirements.txt
    ```

3.  Run the notebook:
    ```bash
    jupyter lab notebooks/modernbert-masked-language-modeling/modernbert-masked-language-modeling.ipynb
    ```

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/modernbert-masked-language-modeling/README.md" />
