# Person Counting System using YoloV8 and OpenVINO

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/person-counting-webcam/person-counting.ipynb)

In this project, we utilized the YOLOv8 Object Counting class to develop a real-time person counting system using the YOLOv8 object detection model and tracking, optimized for Intel's OpenVINO toolkit to enhance inferencing speed. This system effectively monitors the number of individuals entering and exiting a room, leveraging the optimized YOLOv8 model for accurate person detection under varied conditions.

By utilizing the OpenVINO runtime on Intel hardware, the system achieves significant improvements in processing speed, making it ideal for applications requiring real-time data, such as occupancy management and traffic flow control in public spaces and commercial settings.

References:

- YoloV8 Object counting documentation: <a href="https://docs.ultralytics.com/guides/object-counting/" target="_blank">https://docs.ultralytics.com/guides/object-counting/</a>
- OpenVINO Jupyter Notebooks: <a href="https://github.com/openvinotoolkit/openvino_notebooks/" target="_blank">https://github.com/openvinotoolkit/openvino_notebooks/</a>

<div align="center"><img src="person-count.gif" width=900/></div>


## Performance

In this clip, you can see the difference (Inference time and FPS) between running yoloV8 natively with pyTorch vs optimized with OpenVINO.

<div align="center"><img src="optimized.gif" width=900/></div>

## Docker Installation

### Build docker image

```
$ docker build . -t person-count
```

### Run docker container

```
docker run -it --device=/dev/dri --device=/dev/video0 --privileged --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -p 8888:8888 person-count
```

It will prompt the jupyter lab URL on the console, copy and paste it on your browser:

```
Or copy and paste one of these URLs:
        http://localhost:8888/lab?token=<token>
```

## Run it locally

Run the following commands to create a virtual env on your local system

```
python3 -m venv jup1
source jup1/bin/activate
pip install jupyterlab
```

Run jupyter notebook:

```
jupyter lab person-counting.ipynb
```