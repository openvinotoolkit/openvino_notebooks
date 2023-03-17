# 3D Human Pose Estimation with OpenVINO 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks.git/main?labpath=notebooks%2F406-3D-pose-estimation-webcam%2F406-3D-pose-estimation.ipynb)

*Binder is a free service where the webcam will not work, and performance on the video will not be good. For best performance, we recommend installing the notebooks locally.*

![pose estimation_webgl](https://user-images.githubusercontent.com/42672437/183292131-576cc05a-a724-472c-8dc9-f6bc092190bf.gif)

This notebook contains a 3D multi-person pose estimation demo.The model used in this demo is based on [Lightweight OpenPose](https://arxiv.org/abs/1811.12004) and [Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB](https://arxiv.org/abs/1712.03453). It detects 2D coordinates of up to 18 types of keypoints: ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees, and ankles, as well as their 3D coordinates, which could then be used to construct the 3D display of human poses. OpenVINO™ is used to accelerate the inference on multiple devices, such as CPU, GPU and VPU. Also, this 3D display method could be extended to display the inference results of other 3D models without much effort.

## Notebook Contents

This notebook uses the "human-pose-estimation-3d-0001" model from OpenVINO Open Model Zoo, to estimate 3D human pose and represent on a 2D screen. Details of the model can be found [here](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/human-pose-estimation-3d-0001). The input source could be video files or a webcam. It uses the [Three.js](https://pythreejs.readthedocs.io/en/stable/installing.html Python API to display 3D results in a web browser. Note that to display the 3D inference results properly, for Windows and Ubuntu, Chrome is recommended as the web browser. While on macOS, Safari is recommended.

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.

Make sure your [Jupyter extension](https://github.com/jupyter-widgets/pythreejs#jupyterlab) is working properly.
To avoid errors that may arise from the version of the dependency package, we recommend using the **Jupyterlab** instead of the Jupyter notebook to display image results.
```
- pip install --upgrade pip && pip install -r requirements.txt
- jupyter labextension install --no-build @jupyter-widgets/jupyterlab-manager
- jupyter labextension install --no-build jupyter-datawidgets/extension
- jupyter labextension install jupyter-threejs
- jupyter labextension list
```

You should see:
```
JupyterLab v...
  ...
    jupyterlab-datawidgets v... enabled OK
    @jupyter-widgets/jupyterlab-manager v... enabled OK
    jupyter-threejs v... enabled OK
```

