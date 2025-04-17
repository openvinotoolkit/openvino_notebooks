# Text-to-Speech synthesis using Kokoro and OpenVINO

Kokoro is an TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. More details about model can be found in [model card](https://huggingface.co/hexgrad/Kokoro-82M) and [original repository](https://github.com/hexgrad/kokoro)

In this tutorial, we consider how to run Kokoro using OpenVINO.

## Notebook Contents

The tutorial consists of the following steps:

* Download and run Kokoro pipeline
* Convert model to OpenVINO Intermediate Representation (IR) format
* Run Text-to-Speech synthesis using the OpenVINO model
* Interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

For running this notebook, please make sure that `espeak-ng` installed:

**Linux**
```bash
apt-get -qq -y install espeak-ng > /dev/null 2>&1
```
**MacOS**
```bash
brew install espeak-ng
```
**Windows**
1. Go to [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)
2. Click on Latest release
3. Download the appropriate *.msi file (e.g. espeak-ng-20191129-b702b03-x64.msi
4. Run the downloaded installer
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/kokoro/README.md" />
