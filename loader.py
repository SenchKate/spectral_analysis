import zipfile
import requests
import io
from PIL import Image
import sys
import numpy as np
import pandas as pd

import imageio
import os
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float32
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
    classification_report,
)

import cv2

import json

from torchvision import transforms
from efficientnet_pytorch import EfficientNet



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
NUM_EPOCHS = 10
MODEL_SAVE_PATH = '/content/drive/MyDrive/model/model.npy'



def get_transform():
  std = [
     x / 255
     for x in [
         52.42854549,
         41.13263869,
         35.29470731,
         35.12547202,
         32.75119418,
         39.77189372,
         50.80983189,
         53.91031257, 
         21.51845906,
         0.54159901,
         56.63841871,
         42.25028442,
         60.01180004,
     ]
]

  mean = [
      x / 255
      for x in [
          91.94472713,
          74.57486138,
          67.39810048,
          58.46731632,
          72.24985416,
          114.44099918,
          134.4489474,
          129.75758655,
          41.61089189,
          0.86983654,
          101.75149263,
          62.3835689,
          145.87144681,
      ]
  ]
  return transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)]
  )

def get_inverse_transform():
  std = [
     x / 255
     for x in [
         52.42854549,
         41.13263869,
         35.29470731,
         35.12547202,
         32.75119418,
         39.77189372,
         50.80983189,
         53.91031257, 
         21.51845906,
         0.54159901,
         56.63841871,
         42.25028442,
         60.01180004,
     ]
]

  mean = [
      x / 255
      for x in [
          91.94472713,
          74.57486138,
          67.39810048,
          58.46731632,
          72.24985416,
          114.44099918,
          134.4489474,
          129.75758655,
          41.61089189,
          0.86983654,
          101.75149263,
          62.3835689,
          145.87144681,
      ]
  ]  
  std_inv = 1 / (torch.as_tensor(std) + 1e-7)
  mean_inv = -torch.as_tensor(mean) * std_inv
  return transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize(mean_inv, std_inv)]
  )


class EurosatDataset(torch.utils.data.Dataset):
    """Eurosat dataset"""

    def __init__(
        self,
        train,
        root_dir="/content/drive/MyDrive/model/data/ds/images/remote_sensing/otherDatasets/sentinel_2/tif",
        transform=get_transform(),
        seed=42,
    ):
        """
        Args:
            train (bool): If true returns training set, else test
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            seed (int): seed used for train/test split
        """
        #self.seed = seed
        self.size = [64, 64]
        self.num_channels = 13
        self.num_classes = 10
        self.root_dir = root_dir
        self.transform = transform
        self.test_ratio = 0.1
        self.train = train
        self.N = 27000
        self._load_data()

    def _normalize_to_0_to_1(self, img):
        """Normalizes the passed image to 0 to 1

        Args:
            img (np.array): image to normalize

        Returns:
            np.array: normalized image
        """
        img = img + np.minimum(0, np.min(img))  # move min to 0
        img = img / np.max(img)  # scale to 0 to 1
        return img

    def _load_data(self):
        """Loads the data from the passed root directory. Splits in test/train based on seed. By default resized to 256,256"""
        images = np.zeros([self.N, self.size[0], self.size[1], 13], dtype="uint8")
        labels = []
        filenames = []

        i = 0
        # read all the files from the image folder
        for item in tqdm(os.listdir(self.root_dir)):
            f = os.path.join(self.root_dir, item)
            if os.path.isfile(f):
                continue
            for subitem in os.listdir(f):
                sub_f = os.path.join(f, subitem)
                filenames.append(sub_f)
                # a few images are a few pixels off, we will resize them
                image = imageio.imread(sub_f)
                if image.shape[0] != self.size[0] or image.shape[1] != self.size[1]:
                    image = resize(
                        image, (self.size[0], self.size[1]), anti_aliasing=True
                    )
                images[i] = img_as_ubyte(self._normalize_to_0_to_1(image))
                i += 1
                labels.append(item)

        labels = np.asarray(labels)
        filenames = np.asarray(filenames)

        # sort by filenames
        images = images[filenames.argsort()]
        labels = labels[filenames.argsort()]

        # convert to integer labels
        le = preprocessing.LabelEncoder()
        le.fit(np.sort(np.unique(labels)))
        labels = le.transform(labels)
        labels = np.asarray(labels)
        self.label_encoding = list(le.classes_)
        self.data = images
        self.targets = labels  # remember label encoding
    """
        # split into a train and test set as provided data is not presplit
        X_train, X_test, y_train, y_test = train_test_split(
            images,
            labels,
            test_size=self.test_ratio,
            #random_state=self.seed,
            stratify=labels,
        )
    """
        #if self.train:
            

        #else:
         #   self.data = X_test
          #  self.targets = y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx]

        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]



class AllowEmptyClassImageFolder(ImageFolder):
    '''Subclass of ImageFolder, which only finds non-empty classes, but with their correct indices given other empty
    classes. This counter-acts the changes in torchvision 0.10.0, in which DatasetFolder does not allow empty classes
    anymore by default. Versions before 0.10.0 do not expose `find_classes`, and thus this change does not change the
    functionality of `ImageFolder` in earlier versions.
    '''
    def find_classes(self, directory):
        with os.scandir(directory) as scanit:
            class_info = sorted((entry.name, len(list(os.scandir(entry.path)))) for entry in scanit if entry.is_dir())
        class_to_idx = {class_name: index for index, (class_name, n_members) in enumerate(class_info) if n_members}
        if not class_to_idx:
            raise FileNotFoundError(f'No non-empty classes found in \'{directory}\'.')
        return list(class_to_idx), class_to_idx




class Jakob_Loaders():
  def __init__(self, root):
    self.root = root

  def get_transform_rsicd(self):
        dataloader= torch.utils.data.DataLoader(
                          torchvision.datasets.ImageFolder(self.root,
                                                    transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
          ])
                          ),
                          batch_size=1, shuffle=False,
                          num_workers=1)
        return dataloader



  def get_transform_mnist(self):
    dataloader= torch.utils.data.DataLoader(
                      torchvision.datasets.ImageFolder(self.root,
                                                transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.3201, 0.3182, 0.3629), (0.1804, 0.3569, 0.1131)),
      ])
                      ),
                      batch_size=1, shuffle=False,
                      num_workers=1)
    return dataloader

  def get_transform_sen12ms(self):
      dataloader= torch.utils.data.DataLoader(
                        torchvision.datasets.ImageFolder(self.root,
                                                  transforms.Compose([
              transforms.Resize((224, 224)),
              transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize((0.1674, 0.1735, 0.2059), (0.1512, 0.1152, 0.1645)),
        ])
                        ),
                        batch_size=1, shuffle=False,
                        num_workers=1)
      return dataloader
     
  def get_tranform_cifar(self):
    dataloader= torch.utils.data.DataLoader(
                  torchvision.datasets.ImageFolder(self.root,
                                            transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
                  ),
                  batch_size=1, shuffle=False,
                  num_workers=1)
    return dataloader

  def get_tranform_xView2(self):
      dataloader= torch.utils.data.DataLoader(
                    torchvision.datasets.ImageFolder(self.root,
                                              transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
                    ),
                    batch_size=1, shuffle=False,
                    num_workers=1)
      return dataloader

  def get_tranform_So2SatLCZ4(self):
          dataloader= torch.utils.data.DataLoader(
                        torchvision.datasets.ImageFolder(self.root,
                                                  transforms.Compose([
              transforms.Resize((224, 224)),
              transforms.CenterCrop(224),
          transforms.ToTensor(),
         transforms.Normalize((0.2380, 0.3153, 0.5004), (0.0798, 0.1843, 0.0666)),
        ])
                        ),
                        batch_size=1, shuffle=False,
                        num_workers=1)
          return dataloader
    

  def get_loader_Sun(self):
    val_ood_loader = torch.utils.data.DataLoader(
                  torchvision.datasets.ImageFolder(self.root,
                                                  transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
), batch_size=1, shuffle=False,
                  num_workers=1)
    return val_ood_loader