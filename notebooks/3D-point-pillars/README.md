# **PointPillar for 3D object detection**

PointPillar is a fast and efficient deep-learning architecture for 3D object detection from LiDAR point clouds, commonly used in autonomous driving.

Instead of operating directly on raw points or dense 3D voxels, PointPillar groups points into vertical columns ("pillars") and encodes per-pillar features. These pillar features are arranged into a pseudo-image that a 2D convolutional backbone can process. The pipeline is lightweight and well-suited for real-time inference.

Core stages:
- Voxelization / Pillarization: group points into pillars and compute per-pillar statistics.
- Pillar feature encoding: a small network encodes points in each pillar into a fixed-size feature vector.
- Scatter to pseudo-image: place each pillar's feature into a 2D grid (pseudo-image) based on the pillar's X-Y location.
- 2D backbone + neck: apply 2D convolutions to produce multi-scale feature maps.
- Detection head: predict class scores, bounding box regressions, and directions on the pseudo-image.
- Post-processing: decode boxes, apply non-maximum suppression (NMS), and output final detections.

In this tutorial we consider how to run PointPillars with OpenVINO.

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Build extensions
- Exporting the model for OpenVINO
- Run OpenVINO model inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/3D-point-pillars/README.md" />
