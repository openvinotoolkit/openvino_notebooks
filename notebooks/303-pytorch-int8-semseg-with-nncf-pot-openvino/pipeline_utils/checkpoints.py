import os
from os import path as osp
from shutil import copyfile

import torch


def make_additional_checkpoints(checkpoint_path, is_best, epoch, checkpoint_save_dir):
    if is_best:
        best_path = osp.join(checkpoint_save_dir, '{}_best.pth'.format('model'))
        copyfile(checkpoint_path, best_path)
    intermediate_checkpoint = osp.join(checkpoint_save_dir,
                                       'epoch_{}.pth'.format(epoch))
    copyfile(checkpoint_path, intermediate_checkpoint)


def save_checkpoint(model, checkpoint_save_dir):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - compression_ctrl (``PTCompressionAlgorithmController``): The controller containing compression state to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - compression_scheduler: The compression scheduler associated with the model
    - config: Model config".

    Returns:
        The path to the saved checkpoint.
    """
    save_dir = checkpoint_save_dir

    assert os.path.isdir(
        save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    checkpoint_path = os.path.join(save_dir, 'unet') + "_last.pth"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path
