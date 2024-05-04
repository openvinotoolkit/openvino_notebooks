# Kosmos-2: Multimodal Large Language Model and OpenVINO

[KOSMOS-2](https://github.com/microsoft/unilm/tree/master/kosmos-2) is a multimodal large language model (MLLM) that has new capabilities of multimodal grounding and 
referring. KOSMOS-2 can understand multimodal input, follow instructions, 
perceive object descriptions (e.g., bounding boxes), and ground language to the visual world.

Multimodal Large Language Models (MLLMs) have successfully played a role as a general-purpose interface across a wide 
range of tasks, such as language, vision, and vision-language tasks. MLLMs can perceive general modalities, including 
texts, images, and audio, and generate responses using free-form texts under zero-shot and few-shot settings. 

[In this work](https://arxiv.org/abs/2306.14824), authors unlock the grounding capability for multimodal large 
language models. Grounding capability 
can provide a more convenient and efficient human-AI interaction for vision-language tasks. It enables the user to point
 to the object or region in the image directly rather than input detailed text descriptions to refer to it, the model 
 can understand that image region with its spatial locations. Grounding capability also enables the model to respond 
 with visual answers (i.e., bounding boxes), which can support more vision-language tasks such as referring expression 
 comprehension. Visual answers are more accurate and resolve the coreference ambiguity compared with text-only 
 responses. In addition, grounding capability can link noun phrases and referring expressions in the generated free-form 
 text response to the image regions, providing more accurate, informational, and comprehensive answers.


![image](https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/annotated_snowman.jpg)

## Notebook contents
- Prerequisites
- Infer the original model
- Convert the model to OpenVINO IR
- Inference
- Interactive inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).