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

# from torchvision.models.detection.mask_rcnn import MaskRCNN
import torchvision.models.detection as detection  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision.models as models

import torch.distributed as dist
import WeedDataset as WD


# save path for models and checkpoints
save_path = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/model'
os.makedirs(save_path, exist_ok=True)


# Set up WandB
# TODO change to an agkelpie account?
wandb.init(project='weed-detection-refactor1', entity='doriantsai')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu') # for debugging purposes, only use on very small datasets

# setup dataset:
ann_file = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/dataset.json'
img_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/annotated_images'
mask_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/masks'
config_file = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/config/classes.json'
tform = WD.Compose([WD.Rescale(1024),
                WD.RandomBlur(5, (0.5, 2.0)),
                WD.RandomHorizontalFlip(0),
                WD.RandomVerticalFlip(0),
                WD.ToTensor()])

WeedData = WD.WeedDataset(annotation_filename=ann_file,
                       img_dir=img_dir,
                       transforms=tform,
                       mask_dir=mask_dir,
                       config_file=config_file)

# TODO input parameters like James?
batch_size = 10
num_workers = 10
learning_rate = 0.01 # 0.002
momentum = 0.9 # 0.8
weight_decay = 0.0005
num_epochs = 100
step_size = 3 # round(num_epochs / 2)
shuffle = True
rescale_size = int(1024)


# Set up the loss function
def collate_fn(batch):
    return tuple(zip(*batch))

dataloader = DataLoader(WeedData, 
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        collate_fn=collate_fn)


# set up the model
# TODO load from WeedModel?

# Load pre-trained Mask R-CNN model
weights= detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.maskrcnn_resnet50_fpn(weights=weights)

# Replace the last layer to fit the number of classes in your custom dataset
num_classes = 2  # Change this according to your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Replace the mask predictor with a new one
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
model.train()
model.to(device)

# setup the optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
clip_value = 5 # c;o[ gradoemts tp [revemt mams]]


# TODO add validation/evaluation to the training loop, validate every X epochs or so
# TODO add early stopping

# Train the model
start_time = time.time()
print('begin training')
for epoch in range(num_epochs):
    
    print(f'epoch {epoch}/{num_epochs}')
    for i, batch in enumerate(dataloader):
        
        images, targets = batch
        # put all onto the GPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
        # Log training loss to Weights and Biases
        wandb.log({'epoch': epoch+1, 'training_loss': losses.item()})

    
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