#! /usr/bin/env python

"""
create test/train/val dataset and dataloader objects, save them assumes

split_image_data.py has already been run to have images in test/train/val image
folders

assumes data augmented images and negative images have already been added to
said folders as well

based on split_dataset.py from the weed_detection folder (the latter of which is
now depracated)
"""
import WeedDataset as WD
import os
import torch
import pickle


def create_dataset_dataloader(root_dir,
                              json_file,
                              transforms,
                              hp):
    # assume tforms already defined outside of this function
    batch_size = hp['batch_size']
    num_workers = hp['num_workers']
    shuffle= hp['shuffle']

    dataset = WD.WeedDataset(root_dir, json_file, transforms)
    # setup dataloaders for efficient access to datasets
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             collate_fn=dataset.collate_fn)
    return dataset, dataloader


# set hyper parameters of dataset
batch_size = 10
num_workers = 10
learning_rate = 0.005
momentum = 0.9
weight_decay = 0.0001
num_epochs = 100
step_size = round(num_epochs / 2)
shuffle = True
rescale_size = 2056

# make a hyperparameter dictionary
hp={}
hp['batch_size'] = batch_size
hp['num_workers'] = num_workers
hp['learning_rate'] = learning_rate
hp['momentum'] = momentum
hp['step_size'] = step_size
hp['weight_decay'] = weight_decay
hp['num_epochs'] = num_epochs
hp['shuffle'] = shuffle
hp['rescale_size'] = rescale_size

hp_train = hp
hp_test = hp
hp_test['shuffle'] = False

dataset_name = 'Tussock_v1'

root_dir = os.path.join('/home', 'dorian', 'Data', 'AOS_TussockDataset', 'Tussock_v1')
train_folder = os.path.join(root_dir, 'Images', 'Train')
val_folder = os.path.join(root_dir, 'Images', 'Validation')
test_folder = os.path.join(root_dir, 'Images', 'Test')

annotation_dir = os.path.join(root_dir, 'Annotations')
annotations_train = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_train.json')
annotations_val = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_val.json')
annotations_test = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_test.json')

tform_train = WD.Compose([WD.Rescale(rescale_size),
                          WD.RandomBlur(5, (0.5, 2.0)),
                          WD.RandomHorizontalFlip(0.5),
                          WD.RandomVerticalFlip(0.5),
                          WD.ToTensor()])
tform_test = WD.Compose([WD.Rescale(rescale_size),
                         WD.ToTensor()])
# tform_val = tform_test

# now, create the dataset objects with their respective folders/json files
ds_train, dl_train = create_dataset_dataloader(train_folder,
                                               annotations_train,
                                               tform_train,
                                               hp_train)
ds_test, dl_test = create_dataset_dataloader(test_folder,
                                               annotations_test,
                                               tform_test,
                                               hp_test)
ds_val, dl_val = create_dataset_dataloader(val_folder,
                                               annotations_val,
                                               tform_test,
                                               hp_test)

# save datasets for later use (eg, training, etc)
save_dataset_folder = os.path.join('dataset', dataset_name)
os.makedirs(save_dataset_folder, exist_ok=True)
save_dataset_path = os.path.join(save_dataset_folder, dataset_name + '.pkl')
with open(save_dataset_path, 'wb') as f:
    pickle.dump(ds_train, f)
    pickle.dump(ds_test, f)
    pickle.dump(ds_val, f)
    pickle.dump(dl_train, f)
    pickle.dump(dl_test, f)
    pickle.dump(dl_val, f)
    pickle.dump(hp_train, f)
    pickle.dump(hp_test, f)

print('dataset_name: {}'.format(dataset_name))
print('dataset saved as: {}'.format(save_dataset_path))