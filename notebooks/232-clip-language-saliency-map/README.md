# Language-Visual Saliency with CLIP and OpenVINO™

The notebook will cover the following topics:

* Explanation of a _saliency map_ and how it can be used.
* Overview of the CLIP neural network and its usage in generating saliency maps.
* How to split a neural network into parts for separate inference.
* How to speed up inference with OpenVINO™ and asynchronous execution.

## Saliency Map

A saliency map is a visualization technique that highlights regions of interest in an image. For example, it can be used to [explain image classification predictions](https://academic.oup.com/mnras/article/511/4/5032/6529251#389668570) for a particular label. Here is an example of a saliency map that we will get in this notebook:

![](saliency_map_clip.JPG)