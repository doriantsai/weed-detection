#! /usr/bin/env python3

"""
Train MaskRCNN

A class to train the MaskRCNN model on weed data

A #NOTE on the hyper parameters
Batch size was set to 10 to max out the available GPU RAM, 
although input image size and number of workers will also influence this. 
The number of epochs for training was set to 100, 
as an arbitrarily long time to satisfy convergence with the intention of plotting out the learning and loss curves. 
Learning rate, momentum, and weight_decay were initially taken from [Olsen 2019 DeepWeeds], 
and then empirically fine-tuned for learning. 
patience is the number of epochs that remain arbitrarily low or approximately constant, before early stopping occurs
validation_epoch_frequency is the frequency of performing network validation

This was sufficient to develop a pipeline and produce a basic demonstrator. 
Further hyperparameter tuning and sweeps may yield incrementally-improved performance. 
hp = json.load(open(hyper_param_file))['hyper_parameters']
self.num_epochs = hp['num_epochs']                      
self.step_size = hp['step_size'] 
self.rescale_size = int(hp['rescale_size']) # 1024
self.batch_size = hp['batch_size']
self.num_workers = hp['num_workers']
self.learning_rate = hp['learning_rate'] # 0.002
self.momentum = hp['momentum'] # 0.9 # 0.8
self.weight_decay = hp['weight_decay'] # 0.0005
self.patience = hp['patience']
"""

import os
import torch
import wandb
import time
import json

import torch.nn as nn

import torchvision.models.detection as detection  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import weed_detection.WeedDataset as WD
from weed_detection.Annotations import Annotations as Ann


class TrainMaskRCNN:

    # annotation data defaults, a list of dictionaries, defining the
    # annotation file, correesponding image directory and mask directory
    ANNOTATION_DATA_DEFAULT = {'annotation_file': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/dataset.json',
                               'image_dir': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/annotated_images',
                               'mask_dir': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/masks'}
    
    # default output directory, where models and progress checkpoints are saved
    OUTPUT_DIR_DEFAULT = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/model'

    # default config file for class names and colours
    CLASSES_CONFIG_DEFAULT = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/config/classes.json'

    # default image list text files for training, validation and testing:
    IMAGELIST_FILES_DEFAULT = {'train_file': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/metadata/train.txt',
                               'val_file': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/metadata/val.txt',
                               'test_file': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/metadata/test.txt'}
    
    # default hyper parameter text file for training the model
    HYPER_PARAMETERS_DEFAULT = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/config/hyper_parameters.json'

    # Weights And Biases Project Name & User
    WANDB_PROJECT_NAME_DEFAULT = 'weed-detection-refactor1'
    WANDB_USER_DEFAULT = 'doriantsai'


    def __init__(self,
                 annotation_data: dict=ANNOTATION_DATA_DEFAULT,
                 train_val_ratio: tuple=(0.7, 0.15),
                 imagelist_files: dict=IMAGELIST_FILES_DEFAULT,
                 output_dir: str=OUTPUT_DIR_DEFAULT,
                 classes_config_file: str=CLASSES_CONFIG_DEFAULT,
                 hyper_param_file: str=HYPER_PARAMETERS_DEFAULT,
                 wandb_project_name: str=WANDB_PROJECT_NAME_DEFAULT,
                 wandb_user: str=WANDB_USER_DEFAULT):
        """_summary_

        Args:
            annotation_data (dict, optional): annotation data that points to the annotation file, image directory and mask directory. Defaults to ANNOTATION_DATA_DEFAULT.
            train_val_ratio (tuple, optional): a ratio of how many images to use for train/validation/testing. Defaults to (0.7, 0.15).
            imagelist_files (dict, optional): a trio of absolute filepaths to imagelists textfiles for training/validation/testing. Defaults to IMAGELIST_FILES_DEFAULT.
            output_dir (str, optional): absolute filepath to output directory to save the model. Defaults to OUTPUT_DIR_DEFAULT.
            classes_config_file (str, optional): absolute filepath to class config file. Defaults to CLASSES_CONFIG_DEFAULT.
            hyper_param_file (str, optional): absolute filepath to hyperparameters file. Defaults to HYPER_PARAMETERS_DEFAULT.
            wandb_project_name (str, optional): name that wandb project. Defaults to WANDB_PROJECT_NAME_DEFAULT.
            wandb_user (str, optional): username for wandb. Defaults to WANDB_USER_DEFAULT.
        """

        # create annotation set from annotation data
        self.annotation_data = annotation_data
        self.annotation_object = Ann(filename=annotation_data['annotation_file'],
                                     img_dir=annotation_data['image_dir'],
                                     mask_dir=annotation_data['mask_dir'])
    
        # loop to check all mask shapes
        # for i, mask_name in enumerate(self.annotation_object.masks):
        #     mask_path = os.path.join(annotation_data['mask_dir'], mask_name)
        #     mask =  np.array(PILImage.open(mask_path))
        #     print(f'{i}: mask size = {mask.shape}')

        self.num_classes = self.annotation_object.num_classes + 1 # for background/negative

        # handle train_val_ratio
        self.train_val_ratio = train_val_ratio

        # class config file for consistent colours/plots across different functions
        self.classes_config_file = classes_config_file

        # imagelist text files:
        self.imagelist_files = imagelist_files

        # read config file for hyper parameters, see comments at top of file for descriptions
        hp = json.load(open(hyper_param_file))['hyper_parameters']
        self.num_epochs = hp['num_epochs']
        self.step_size = hp['step_size'] 
        self.rescale_size = int(hp['rescale_size']) # 1024
        self.batch_size = hp['batch_size']
        self.num_workers = hp['num_workers']
        self.learning_rate = hp['learning_rate'] # 0.002
        self.momentum = hp['momentum'] # 0.9 # 0.8
        self.weight_decay = hp['weight_decay'] # 0.0005
        self.patience = hp['patience']
        self.validation_epoch_frequency = hp['validation_epoch_frequency']

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Set up WandB
        wandb.init(project=wandb_project_name, entity=wandb_user) # TODO change to an agkelpie account?

        # save path for models and checkpoints
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # create model
        self.model = self.create_model()
        # end init


    def process_train_val_test_ratio(self, train_val_ratio):
        """process_train_val_test_ratio
        given a train_val ratio, compute the remaining test ratio and then given
        the number of images from self.annotation_object.imgs, compute the
        precise number of images for each set

        process train_val ratio for the percentages of data for training, validation and testing, respectively
        input: tuple of two numbers,
        output: tuple of 3 numbers corresponding to the number of images for training/validation/testing

        Args:
            train_val_ratio (tuple of floats): tuple 

        Returns:
            tuple of ints: number of train, val, test images
        """
        assert isinstance(train_val_ratio, tuple), "train_val_ratio should be a tuple"
        assert len(train_val_ratio) == 2, "train_val ratio should have a length of 2"
        assert train_val_ratio[0] > 0, "train should be greater than zero"
        assert train_val_ratio[1] >= 0, "val should be greater or equal to zero"
        assert train_val_ratio[0] + train_val_ratio[1] <= 1, "sum should be less than or equal to 1"

        # split data into training/validation based off of text files
        train_ratio = train_val_ratio[0]
        val_ratio = train_val_ratio[1]
        test_ratio = 1.0 - train_ratio - val_ratio

        if train_ratio > 1 or train_ratio < 0:
            ValueError(train_ratio,f'train_ratio must 0 < train_ratio <= 1, train_ratio = {train_ratio}')
        if val_ratio < 0 or val_ratio >= 1:
            ValueError(val_ratio, f'val_ratio must be 0 < val_ratio < 1, val_ratio = {val_ratio}')
        if test_ratio < 0 or test_ratio >= 1:
            ValueError(test_ratio, f'0 < test_ratio < 1, test_ratio = {test_ratio}')
        if not ((train_ratio + val_ratio + test_ratio) == 1):
            ValueError(train_ratio, 'sum of train/val/test ratios must equal 1')

        n_img = len(self.annotation_object.imgs)
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

        return (n_train, n_val, n_test)


    def split_data(self, train_file: str, val_file: str, test_file: str):
        """split data
        Split data into train/val/test sets and output image lists as text files

        Args:
            data_filenames (str): absolute filepaths for saving output
        """

        # split data
        n_train, n_val, n_test = self.process_train_val_test_ratio(self.train_val_ratio)

        # randomly split the images (might use pytorch random split of images?)
        # TODO make robust to 0 n_test
        train_val_filenames, test_filenames = train_test_split(self.annotation_object.imgs, test_size=int(n_test), random_state=42) # hopefully works with 0 as test_ratio?
        train_filenames, val_filenames = train_test_split(train_val_filenames, test_size=int(n_val), random_state=42)

        # sanity check:
        print(f'length of train_filenames = {len(train_filenames)}')
        print(f'length of val_filenames = {len(val_filenames)}')
        print(f'length of test_filenames = {len(test_filenames)}')

        # save filename lists to textfiles for future use
        self.annotation_object.generate_imagelist_txt(train_file, train_filenames)
        self.annotation_object.generate_imagelist_txt(val_file, val_filenames)
        self.annotation_object.generate_imagelist_txt(test_file, test_filenames)
        print('successful split data')


    def create_datasets(self, annotation_data: dict, classes_config_file: str, train_file: str, val_file: str, test_file: str):
        """create_datasets
        Create pytorch dataset objects for batched training, validation and testing

        A PyTorch Dataset object is used to represent a collection of data that
        can be iterated over, typically in batches, during training or
        evaluation of a machine learning model. It provides an easy way to load,
        preprocess, and transform the data before feeding it to a machine
        learning model, and allows for data augmentation and customization.
        Using PyTorch Dataset objects ensures efficient handling of large
        datasets and loading only the required data into memory, making it
        easier to train machine learning models.

        Args:
            annotation_data (dict): dict to annotation json, image_dir, mask_dir
            classes_config_file (str): absolute filepath to classes config
            train_file (str): absolute filepath to training images textfile
            val_file (str): absolute filepath to validation images textfile
            test_file (str): absolute filepath to testing images textfile

        Returns:
            pytorch datasets: datasets for training, validation and testing
        """        
        # setup training and testing transforms
        tform_train = WD.Compose([WD.Rescale(1024),
                WD.RandomBlur(5, (0.5, 2.0)),
                WD.RandomHorizontalFlip(0),
                WD.RandomVerticalFlip(0),
                WD.ToTensor()])
        # just rescaling and convert to tensor for validation (we don't want randomisation)
        tform_val = WD.Compose([WD.Rescale(1024),
                WD.ToTensor()])
        
        WeedDataTrain = WD.WeedDataset(annotation_filename=annotation_data['annotation_file'],
                       img_dir=annotation_data['image_dir'],
                       transforms=tform_train,
                       mask_dir=annotation_data['mask_dir'],
                       classes_file=classes_config_file,
                       imgtxt_file=train_file)

        WeedDataVal = WD.WeedDataset(annotation_filename=annotation_data['annotation_file'],
                       img_dir=annotation_data['image_dir'],
                       transforms=tform_val,
                       mask_dir=annotation_data['mask_dir'],
                       classes_file=classes_config_file,
                       imgtxt_file=val_file)
        
        WeedDataTest = WD.WeedDataset(annotation_filename=annotation_data['annotation_file'],
                       img_dir=annotation_data['image_dir'],
                       transforms=tform_val,
                       mask_dir=annotation_data['mask_dir'],
                       classes_file=classes_config_file,
                       imgtxt_file=test_file)

        return WeedDataTrain, WeedDataVal, WeedDataTest
    

    def create_dataloader(self, WeedData, shuffle: bool=True):
        """create_dataloader
        A PyTorch DataLoader is an iterable that loads data from a PyTorch
        Dataset object in batches during training or evaluation of a machine
        learning model. It provides options for parallel data loading, batching,
        shuffling, and automatic memory pinning for efficient GPU usage. The
        DataLoader is a critical component of the PyTorch data loading pipeline
        and makes it easy to load and process data for deep learning models.

        Args:
            WeedData (pytorch dataset):  dataset from  create_datasets()
            shuffle (bool, optional): shuffle the order of the images in the
            dataset or not. Defaults to True.

        Returns:
            pytorch dataloader: dataloader of the dataset
        """        
        """ create dataloader from dataset"""
        dataloader = DataLoader(WeedData, 
                                batch_size=self.batch_size,
                                shuffle=shuffle,
                                num_workers=self.num_workers,
                                collate_fn=self.collate_fn)
        return dataloader


    def create_model(self):
        """create_model
        Setup neural network model architecture for object detection, MaskRCNN
        for training

        Returns:
            pytorch MaskRCNN model: pretrained/default weights to MaskRCNN
            (pre-trained on COCO)
        """    
        weights= detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = models.detection.maskrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        # Replace the mask predictor with a new one
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, self.num_classes)
        model.to(self.device)
        return model


    def train_pipeline(self):
        """train_pipeline
        Training pipeline that splits data, creates datasets, trains the model given the datasets
        """        

        # split the data
        self.split_data(self.imagelist_files['train_file'],
                        self.imagelist_files['val_file'],
                        self.imagelist_files['test_file'])
        
        # create datasets from given files
        WeedDataTrain, WeedDataVal, __ = self.create_datasets(self.annotation_data, 
                                                              self.classes_config_file, 
                                                              self.imagelist_files['train_file'], 
                                                              self.imagelist_files['val_file'], 
                                                              self.imagelist_files['test_file'])
        
        # train the model
        self.train_model(WeedDataTrain, WeedDataVal)
        print('successfully completed train_pipeline')


    def train_model(self, 
                    WeedDataTrain, 
                    WeedDataVal, 
                    model_name: str=None, 
                    SAVE_CHECKPOINTS: bool=False):
        """train_model
        Train the model on the training dataset and validate on the validation
        dataset, save checkpoints if desired, save to a given model_name

        Args:
            WeedDataTrain (pytorch dataset): dataset for training
            WeedDataVal (pytorch dataset): dataset for validation
            model_name (str, optional): name of the model for saving. Defaults to None.
            SAVE_CHECKPOINTS (bool, optional): save checkpoints or not. Defaults to False.
        """        

        # set unique dataset-specific model name and output folder
        if model_name is None:
            model_name = self.annotation_object.dataset_name
        os.makedirs(os.path.join(self.output_dir, model_name), exist_ok=True)
        
        # create dataloaders separate for train/val
        dataloader_train = self.create_dataloader(WeedData=WeedDataTrain, shuffle=True)
        dataloader_val = self.create_dataloader(WeedData=WeedDataVal, shuffle=False)

        # setup the optimizer and learning rate scheduler
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=0.1)
        # clip_value = 5 # clip gradients

        start_time = time.time()
        lowest_val = 1e6
        best_epoch = 0 
        counter = 0

        print('begin training')
        for epoch in range(self.num_epochs):
            # print(f'epoch {epoch+1}/{self.num_epochs}')

            # train model for one epoch
            train_loss = self.train_one_epoch(self.model, dataloader_train, optimizer)

            # compute mAP score
            train_mAP = self.compute_mAP(self.model, dataloader_val)

            lr_scheduler.step()

            # validation 
            if (epoch % self.validation_epoch_frequency) == (self.validation_epoch_frequency - 1):
                val_loss = self.validate_epoch(self.model, dataloader_val)

                # compute mAP score
                val_mAP = self.compute_mAP(self.model, dataloader_val)

                # Log training loss to Weights and Biases
                print(f'Epoch: {epoch+1}/{self.num_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}')
                wandb.log({'epoch': epoch+1, 'training_loss': train_loss, 'training_mAP': train_mAP, 'validation_loss': val_loss, 'validation_mAP': val_mAP})

                # Save the model checkpoint after each validation epoch
                checkpoint_path = f'checkpoint_{epoch+1}.pth'
                
                if SAVE_CHECKPOINTS:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, model_name, checkpoint_path))
                wandb.save(checkpoint_path)

                # save the best epoch via min running loss
                if val_loss < lowest_val:
                    lowest_val = val_loss
                    best_epoch = epoch
                    best_name = os.path.join(self.output_dir, model_name, 'model_best.pth')
                    torch.save(self.model.state_dict(), best_name)
                    counter = 0
                else:
                    # Early stopping: whenever we find a new "best" epoch, we
                    # reset the counter otherwise, we increment the counter. If
                    # the counter reaches a threshold, then we trigger early
                    # stopping (stop the training). 
                    counter += 1
                    if counter == self.patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        break
                # end validation section
            else:
                print(f'Epoch: {epoch+1}/{self.num_epochs} | Train loss: {train_loss:.4f}')
                wandb.log({'epoch': epoch+1, 'training_loss': train_loss, 'training_mAP': train_mAP})

            # end training/validation loop

        print('training complete')
        end_time = time.time()
        sec = end_time - start_time
        print('training time: {} sec'.format(sec))
        print('training time: {} min'.format(sec / 60.0))
        print('training time: {} hrs'.format(sec / 3600.0))

        print(f'best epoch: {best_epoch+1}, as {best_name}')

        # save trained model for inference
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, model_name, 'model_maxepochs.pth'))
        print('model saved: {}'.format(self.output_dir))


    def train_one_epoch(self, model, dataloader, optimizer):
        """train_one_epoch
        Code for training the model over one epoch 

        Args:
            model (_type_): _description_
            dataloader (_type_): _description_
            optimizer (_type_): _description_

        Returns:
            _type_: _description_
        """        
        running_train_loss = 0.0
        model.train()
        for i, batch in enumerate(dataloader):
        
            images, targets = batch
            # put all onto the GPU
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_train_loss += losses.item()
            
            optimizer.zero_grad()
            
            losses.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
        return running_train_loss/len(dataloader)


    def validate_epoch(self, model, dataloader):
        """validate_epoch
        Code for running validation on the model during training

        Args:
            model (_type_): _description_
            dataloader (_type_): _description_

        Returns:
            _type_: _description_
        """        
        model.train()

        # freeze the batch normalisation layers, because model.eval() doesn't
        # provide losses
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        running_val_loss = 0.0
        with torch.no_grad():
            for images, targets in dataloader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                running_val_loss += losses.item()
        return running_val_loss/len(dataloader)


    def compute_mAP(self, model, dataloader):
        """compute_mAP
        compute the mean average precision of the model as a measure of
        performance during training/validation

        Args:
            model (_type_): _description_ 
            dataloader (_type_): _description_

        Returns:
            _type_: _description_
        """        

        metric = MeanAveragePrecision()
        # NOTE consider putting into evaluation pipeline?
        model.eval()

        # Define the ground truth and predictions arrays
        for images, targets in dataloader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # make predictions
            with torch.no_grad():
                outputs = model(images)
                metric.update(outputs, targets)

        ans = metric.compute()
        return float(ans['map'])


    # Set up the loss function
    def collate_fn(self, batch):
        return tuple(zip(*batch))
    

if __name__ == "__main__":

    print('TrainMaskRCNN.py')

    TrainMask = TrainMaskRCNN() # rely on defaults
    TrainMask.train_pipeline()

    import code
    code.interact(local=dict(globals(), **locals()))