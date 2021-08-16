"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import functools
import logging
import random
import sys
from pathlib import Path

import numpy
import torch
from torch.backends import cudnn
from tqdm import tqdm

from pipeline_utils import loss_funcs
from pipeline_utils.arg_parser import get_arguments_parser
from pipeline_utils.checkpoints import make_additional_checkpoints
from pipeline_utils.checkpoints import save_checkpoint
from pipeline_utils.dataset import get_dataset
from pipeline_utils.dataset import load_dataset
from pipeline_utils.metric import IoU
from pipeline_utils.optimizer import make_optimizer
from unet import UNet
from unet import center_crop

DEVICE = 'cuda:0'
RESULT_DIR = Path('results')

logger = logging.getLogger("example")

if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    hdl = logging.StreamHandler(stream=sys.stdout)
    hdl.setFormatter(logging.Formatter("%(message)s"))
    hdl.setLevel(logging.INFO)
    logger.addHandler(hdl)


class Train:
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.
    - model_name: Name of the model to be trained - determines model-specific processing
    of the results (i.e. whether center crop should be applied, what outputs should be counted in metrics, etc.)
    """

    def __init__(self, model, data_loader, optim, criterion, metric, device, model_name):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device
        self.model_name = model_name

    def run_epoch(self):
        """Runs an epoch of training.

        Returns:
        - The epoch loss (float).

        """

        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # Forward propagation
            outputs = self.model(inputs)

            outputs_size_hw = (outputs.size()[2], outputs.size()[3])
            labels = center_crop(labels, outputs_size_hw).contiguous()

            import torch
            torch.set_deterministic(False)
            # Loss computation
            loss = self.criterion(outputs, labels)
            torch.set_deterministic(True)

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()


            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of the evaluation metric
            self.metric.add(outputs.detach(), labels.detach())

        return epoch_loss / len(self.data_loader), self.metric.value()


class Test:
    """Tests the ``model`` on the specified test dataset using the
    data loader, and loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to test.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.
    - model_name: Name of the model to be trained - determines model-specific processing
    of the results (i.e. whether center crop should be applied, what outputs should be counted in metrics, etc.)

    """

    def __init__(self, model, data_loader, criterion, metric, device, model_name):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.device = device
        self.model_name = model_name

    def run_epoch(self):
        """Runs an epoch of validation.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        """
        self.model.eval()
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            with torch.no_grad():
                # Forward propagation
                outputs = self.model(inputs)

                outputs_size_hw = (outputs.size()[2], outputs.size()[3])
                labels = center_crop(labels, outputs_size_hw).contiguous()

                # Loss computation
                torch.set_deterministic(False)
                loss = self.criterion(outputs, labels)
                torch.set_deterministic(True)

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            self.metric.add(outputs.detach(), labels.detach())

        return epoch_loss / len(self.data_loader), self.metric.value()


def train(model, train_loader, val_loader, criterion, class_encoding, epochs, checkpoint_save_dir, args):
    logger.info("\nTraining...\n")

    seed = 0
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.set_deterministic(True)

    # Check if the network architecture is correct
    logger.info(model)

    optim_config = {
        "optimizer": {
            "type": "Adam",
            "optimizer_params": {
                "lr": 5.0e-5,
                "weight_decay": 2.0e-4
            },
            "schedule_type": "step",
            "step": 100,
            "gamma": 0.1
        },
    }

    params_to_optimize = model.parameters()
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, optim_config)

    # Evaluation metric

    ignore_index = None
    if ('unlabeled' in class_encoding):
        ignore_index = list(class_encoding).index('unlabeled')

    metric = IoU(len(class_encoding), ignore_index=ignore_index)

    best_miou = -1

    # Start Training
    train_obj = Train(model, train_loader, optimizer, criterion, metric, DEVICE, 'unet')
    val_obj = Test(model, val_loader, criterion, metric, DEVICE, 'unet')

    for epoch in range(0, epochs):
        logger.info(">>>> [Epoch: {0:d}] Training".format(epoch))

        epoch_loss, (iou, miou) = train_obj.run_epoch()
        lr_scheduler.step(epoch)

        logger.info(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                    format(epoch, epoch_loss, miou))

        save_freq = 1
        if (epoch + 1) % save_freq == 0 or epoch + 1 == epochs:
            logger.info(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val_obj.run_epoch()

            logger.info(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                        format(epoch, loss, miou))

            is_best = miou > best_miou
            if is_best:
                best_miou = miou

            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == epochs or is_best:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    logger.info("{0}: {1:.4f}".format(key, class_iou))

            # Save the model if it's the best thus far
            checkpoint_path = save_checkpoint(model, checkpoint_save_dir)
            make_additional_checkpoints(checkpoint_path, is_best, epoch, checkpoint_save_dir)

    return model


def test(model, test_loader, criterion, class_encoding):
    logger.info("\nTesting...\n")

    # Evaluation metric
    ignore_index = None

    if ('unlabeled' in class_encoding):
        ignore_index = list(class_encoding).index('unlabeled')

    metric = IoU(len(class_encoding), ignore_index=ignore_index)

    # Test the trained model on the test set
    test_obj = Test(model, test_loader, criterion, metric, DEVICE, 'unet')

    logger.info(">>>> Running test dataset")

    loss, (iou, miou) = test_obj.run_epoch()

    logger.info(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        logger.info("{0}: {1:.4f}".format(key, class_iou))

    return miou


def main(argv):
    args = get_arguments_parser().parse_args(argv)
    dataset = get_dataset()

    model = UNet(n_classes=len(dataset.color_encoding), input_size_hw=[368, 480]).to(DEVICE)

    if args.resuming_checkpoint_path is not None:
        model.load_state_dict(torch.load(args.resuming_checkpoint_path))

    loaders, w_class = load_dataset(dataset, args.dataset_dir, args.batch_size, logger)
    train_loader, val_loader, init_loader = loaders
    criterion = functools.partial(loss_funcs.cross_entropy, weight=w_class.to(DEVICE))

    if 'export' in args.mode and ('train' not in args.mode and 'test' not in args.mode):
        torch.onnx.export(model, torch.zeros([1, 3, 368, 480]).to(DEVICE), args.to_onnx)
        logger.info("Saved to {}".format(args.to_onnx))
        return

    if 'train' in args.mode:
        train(model, train_loader, val_loader, criterion, dataset.color_encoding, args.epochs, RESULT_DIR, args)
        model.load_state_dict(torch.load(RESULT_DIR / 'model_best.pth'))

    if 'test' in args.mode:
        test(model, val_loader, criterion, dataset.color_encoding)

    if 'export' in args.mode:
        torch.onnx.export(model, torch.zeros([1, 3, 368, 480]).to(DEVICE), args.to_onnx)
        logger.info("Saved to {}".format(args.to_onnx))


if __name__ == '__main__':
    main(sys.argv[1:])
