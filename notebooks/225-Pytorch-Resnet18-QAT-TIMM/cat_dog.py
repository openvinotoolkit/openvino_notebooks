import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np

class Custom_dataset(Dataset):
    def __init__(self,annotation_file,root_dir,transform):
        self.img_label = pd.read_csv(annotation_file)
        self.img_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_label.iloc[idx, 0])
        image = Image.open(img_path).resize((224,224))
        label = self.img_label.iloc[idx,1]
        label = torch.tensor(label)
        if self.transform:
            image = self.transform(image)
        return image, label
