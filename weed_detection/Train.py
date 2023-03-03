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

# set up the model
# TODO load from WeedModel?
model = MaskRCNN(num_classes=2)

# setup the optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# Set up the loss function
def collate_fn(batch):
    return tuple(zip(*batch))

# Train the model
for epoch in range(10):
    for i, batch in enumerate(data_loader):
        images, targets = batch
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Log the loss and learning rate to WandB
        wandb.log({'loss': losses.item(), 'learning_rate': optimizer.param_groups[0]['lr']})

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        lr_scheduler.step()