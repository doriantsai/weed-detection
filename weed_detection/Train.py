#! /usr/bin/env/python3

"""
- train model using pytorch and show progress on WAndB
- MaskRCNN, 
- load dataset/dataloader
- save checkpoints
- early stopping, save training plots & performance
"""

import os
import torch
import torchvision
import wandb

from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from weed_detection import WeedDataset

# Set up WandB
# TODO change to agkelpie?
wandb.init(project='weed-detection-refactor1', entity='doriantsai')



