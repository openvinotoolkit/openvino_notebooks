
import os
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.utils.data as data

from pipeline_utils import data_utils
from pipeline_utils.transforms import get_joint_transforms


class CamVid(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.


    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """
    # Training dataset root folders
    train_folder = 'train'
    train_lbl_folder = 'trainannot'

    # Validation dataset root folders
    val_folder = 'val'
    val_lbl_folder = 'valannot'

    # Test dataset root folders
    test_folder = 'test'
    test_lbl_folder = 'testannot'

    # Images extension
    img_extension = '.png'

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        ('road_marking', (255, 69, 0)),
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        ('unlabeled', (0, 0, 0))
    ])

    def __init__(self,
                 root,
                 image_set='train',
                 transforms=None,
                 loader=data_utils.pil_loader):
        super().__init__()
        self.root_dir = root
        self.mode = image_set
        self.transforms = transforms
        self.loader = loader

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = data_utils.get_files(
                os.path.join(self.root_dir, self.train_folder),
                extension_filter=self.img_extension)

            self.train_labels = data_utils.get_files(
                os.path.join(self.root_dir, self.train_lbl_folder),
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = data_utils.get_files(
                os.path.join(self.root_dir, self.val_folder),
                extension_filter=self.img_extension)

            self.val_labels = data_utils.get_files(
                os.path.join(self.root_dir, self.val_lbl_folder),
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = data_utils.get_files(
                os.path.join(self.root_dir, self.test_folder),
                extension_filter=self.img_extension)

            self.test_labels = data_utils.get_files(
                os.path.join(self.root_dir, self.test_lbl_folder),
                extension_filter=self.img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)

        if self.transforms is not None:
            img, label = self.transforms(img, label)

        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        if self.mode.lower() == 'val':
            return len(self.val_data)
        if self.mode.lower() == 'test':
            return len(self.test_data)

        raise RuntimeError("Unexpected dataset mode. "
                           "Supported modes are: train, val and test")


def get_dataset() -> torch.utils.data.Dataset:
    dataset = CamVid
    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if 'road_marking' in dataset.color_encoding:
        del dataset.color_encoding['road_marking']

    return dataset


def load_dataset(dataset, dataset_dir, batch_size, logger):
    logger.info("\nLoading dataset...\n")

    logger.info("Selected dataset: {}".format('CamVid'))
    logger.info("Dataset directory: {}".format(dataset_dir))

    transforms_train = get_joint_transforms()
    transforms_val = get_joint_transforms()

    # Get selected dataset
    train_set = dataset(
        root=dataset_dir,
        image_set='train',
        transforms=transforms_train)

    val_set = dataset(
        dataset_dir,
        image_set='val',
        transforms=transforms_val)

    train_sampler = None

    # Samplers
    train_sampler = torch.utils.data.RandomSampler(train_set)

    seed = 0
    train_shuffle = train_sampler is None and seed is None
    batch_size = batch_size
    num_workers = 4

    def create_train_data_loader(batch_size_):
        return torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size_,
            sampler=train_sampler, num_workers=num_workers,
            collate_fn=data_utils.collate_fn, drop_last=True,
            shuffle=train_shuffle)
    # Loaders
    train_loader = create_train_data_loader(batch_size)
    init_loader = deepcopy(train_loader)
    val_sampler = torch.utils.data.SequentialSampler(val_set)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1, num_workers=num_workers,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=data_utils.collate_fn, drop_last=False)

    # Get encoding between pixel values in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    logger.info("Number of classes to predict: {}".format(num_classes))
    logger.info("Train dataset size: {}".format(len(train_set)))
    logger.info("Validation dataset size: {}".format(len(val_set)))

    images, labels = iter(train_loader).next()
    logger.info("Image size: {}".format(images.size()))
    logger.info("Label size: {}".format(labels.size()))
    logger.info("Class-color encoding: {}".format(class_encoding))

    class_weights = get_class_weights(train_set, num_classes, logger)

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float()
        # Set the weight of the unlabeled class to 0
        if 'unlabeled' in class_encoding:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    logger.info("Class weights: {}".format(class_weights))

    return (train_loader, val_loader, init_loader), class_weights


def get_class_weights(train_set, num_classes, logger):
    # Get class weights from the selected weighing technique
    logger.info("\nWeighing technique: {}".format('mfb'))

    train_loader_for_weight_count = torch.utils.data.DataLoader(
        train_set,
        batch_size=1, collate_fn=data_utils.collate_fn)
    logger.info("Computing class weights...")
    logger.info("(this can take a while depending on the dataset size)")
    class_weights = data_utils.median_freq_balancing(train_loader_for_weight_count, num_classes)
    return class_weights
