import matplotlib.pyplot as plt
import PIL
import numpy as np


def visualize_results(orig_img:PIL.Image.Image, answer:str, question:str = None):
    """
    Helper function for results visualization

    Parameters:
       orig_img (PIL.Image.Image): original image
       answer (str): model answer in text format.
       question (str, *optional*, None): input question, if not provided answer will be used as caption
    Returns:
       fig (matplotlib.pyplot.Figure): matplotlib generated figure contains drawing result
    """
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.grid(False)
    ax.imshow(np.array(orig_img))
    qa_text = "question: {}\nanswer: {}"
    cap_text = "caption: {}"
    ax.set_title(qa_text.format(question, answer) if question is not None else cap_text.format(answer),
                 y=-0.01, pad=-30 if question is not None else -15)
    return fig
