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
import time
import shutil

# from torchvision.models.detection.mask_rcnn import MaskRCNN
import torch.nn as nn

import torchvision.models.detection as detection  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.model_selection import train_test_split

import torch.distributed as dist
import WeedDataset as WD
import Annotations as Ann


# Set up the loss function
def collate_fn(batch):
    return tuple(zip(*batch))

# save path for models and checkpoints
save_path = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/model'
os.makedirs(save_path, exist_ok=True)

# Set up WandB
wandb.init(project='weed-detection-refactor1', entity='doriantsai') # TODO change to an agkelpie account?

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu') # for debugging purposes, only use on very small datasets

# setup dataset:
ann_file = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/dataset.json'
img_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/annotated_images'
mask_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/masks'
config_file = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/config/classes.json'

# create annotation set:
WeedAnn = Ann.Annotations(ann_file, img_dir, mask_dir)

# split data into training/validation based off of text files
# train/val/test ratio
train_ratio = 0.7
val_ratio = 0.15
# test_ratio is remainder
test_ratio = 1.0 - train_ratio - val_ratio

if train_ratio > 1 or train_ratio < 0:
    ValueError(train_ratio,f'train_ratio must 0 < train_ratio <= 1, train_ratio = {train_ratio}')
if val_ratio < 0 or val_ratio >= 1:
    ValueError(val_ratio, f'val_ratio must be 0 < val_ratio < 1, val_ratio = {val_ratio}')
if test_ratio < 0 or test_ratio >= 1:
    ValueError(test_ratio, f'0 < test_ratio < 1, test_ratio = {test_ratio}')
if not ((train_ratio + val_ratio + test_ratio) == 1):
    ValueError(train_ratio, 'sum of train/val/test ratios must equal 1')
    
# training:
train_file = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/metadata/train.txt'

# validation
if val_ratio > 0:
    val_file = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/metadata/val.txt'

# testing
if test_ratio > 0:
    test_file = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/metadata/test.txt'

n_img = len(WeedAnn.imgs)
n_train = round(n_img * train_ratio)

if val_ratio == 0:
    n_val = 0
elif test_ratio == 0:
    n_val = n_img - n_train
else:
    n_val = round(n_img * val_ratio)

if test_ratio == 0:
    n_test = 0
else:
    n_test = n_img - n_train - n_val

print(f'total images: {n_img}')
print(f'n_train = {n_train}')
print(f'n_val = {n_val}')
print(f'n_test = {n_test}')

# randomly split the images (might use pytorch random split of images?)
# TODO make robust to 0 n_test
train_val_filenames, test_filenames = train_test_split(WeedAnn.imgs, test_size=int(n_test), random_state=42) # hopefully works with 0 as test_ratio?
train_filenames, val_filenames = train_test_split(train_val_filenames, test_size=int(n_val), random_state=42)

# sanity check:
print(f'length of train_filenames = {len(train_filenames)}')
print(f'length of val_filenames = {len(val_filenames)}')
print(f'length of test_filenames = {len(test_filenames)}')

WeedAnn.generate_imagelist_txt(train_file, train_filenames)
WeedAnn.generate_imagelist_txt(val_file, val_filenames)
WeedAnn.generate_imagelist_txt(test_file, test_filenames)

tform_train = WD.Compose([WD.Rescale(1024),
                WD.RandomBlur(5, (0.5, 2.0)),
                WD.RandomHorizontalFlip(0),
                WD.RandomVerticalFlip(0),
                WD.ToTensor()])
tform_test = WD.Compose([WD.Rescale(1024),
                WD.ToTensor()])

WeedDataTrain = WD.WeedDataset(annotation_filename=ann_file,
                       img_dir=img_dir,
                       transforms=tform_train,
                       mask_dir=mask_dir,
                       config_file=config_file,
                       imgtxt_file=train_file)

WeedDataVal = WD.WeedDataset(annotation_filename=ann_file,
                       img_dir=img_dir,
                       transforms=tform_test,
                       mask_dir=mask_dir,
                       config_file=config_file,
                       imgtxt_file=val_file)

# TODO input parameters like James?
batch_size = 10
num_workers = 10
learning_rate = 0.002 # 0.002
momentum = 0.9 # 0.8
weight_decay = 0.0005
num_epochs = 100
step_size = 10 # round(num_epochs / 2)
rescale_size = int(1024)

dataloader_train = DataLoader(WeedDataTrain, 
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        collate_fn=collate_fn)

dataloader_val = DataLoader(WeedDataVal, 
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        collate_fn=collate_fn)

# set up the model
# TODO load from WeedModel?

# Load pre-trained Mask R-CNN model
weights= detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.maskrcnn_resnet50_fpn(weights=weights)

# Replace the last layer to fit the number of classes in your custom dataset
num_classes = 2  # TODO Change this according to your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Replace the mask predictor with a new one
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
model.to(device)

# setup the optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
clip_value = 5 # clip gradients

# TODO add early stopping
# Train the model
start_time = time.time()
val_epoch = 2 # validation epoch frequency
lowest_val = 1e6
best_epoch = 0 
print('begin training')
for epoch in range(num_epochs):
    
    # training TODO can make this a function "train_one" epoch
    print(f'epoch {epoch+1}/{num_epochs}')
    running_train_loss = 0.0
    model.train()
    for i, batch in enumerate(dataloader_train):
        
        images, targets = batch
        # put all onto the GPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        running_train_loss += losses.item()
        
        optimizer.zero_grad()
        
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
    # validation 
    if (epoch % val_epoch) == (val_epoch - 1):
        # put into eval mode, NOTE: MaskRCNN outputs predictions in eval mode, so to retain losses:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        running_val_loss = 0.0
        with torch.no_grad():
            
            for i, batch in enumerate(dataloader_val):
                images, targets = batch
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                running_val_loss += losses.item()
                
                # TODO implement these functions
                # total_correct += calculate_correct_predictions(outputs, targets)
                # total_predicted += calculate_predicted_objects(outputs, num_classes)
                # total_gt += calculate_gt_objects(targets, num_classes)
        
        # save the best epoch via min running loss
        if running_val_loss < lowest_val:
            lowest_val = running_val_loss
            best_epoch = epoch
    
        # Log training loss to Weights and Biases
        # wandb.log({'epoch': epoch+1, 'validation_loss': running_val_loss}) 

    # Log training loss to Weights and Biases
    wandb.log({'epoch': epoch+1, 'training_loss': running_train_loss, 'validation_loss': running_val_loss})

    lr_scheduler.step()
    
     # Save the model checkpoint after each epoch
    checkpoint_path = f'checkpoint_{epoch+1}.pth'
    torch.save(model.state_dict(), os.path.join(save_path, checkpoint_path))
    wandb.save(checkpoint_path)
    
    # Check for early stopping
    # if early_stopping.check(loss):
    #     print("Early stopping criterion met")
    #     break
    
    torch.cuda.empty_cache()

print('training done')
end_time = time.time()
sec = end_time - start_time
print('training time: {} sec'.format(sec))
print('training time: {} min'.format(sec / 60.0))
print('training time: {} hrs'.format(sec / 3600.0))

# save trained model for inference
torch.save(model.state_dict(), os.path.join(save_path, 'model_final.pth'))
print('model saved: {}'.format(save_path))

print('done')    
import code
code.interact(local=dict(globals(), **locals()))