#! /usr/bin/env python
"""
randomly split image dataset into training/validation/testing sets
perhaps this should be a function in the SerratedTussockDataset object
"""

import os
import json
import pickle
from SerratedTussockDataset import SerratedTussockDataset, RandomHorizontalFlip, Rescale, ToTensor, Blur, Compose

import torch
import numpy as np

def collate_fn(batch):
    return tuple(zip(*batch))


def split_dataset(root_dir,
                  json_file,
                  hp,
                  dataset_name,
                  dataset_lengths):

    rescale_size = hp['rescale_size']
    batch_size = hp['batch_size']
    num_workers = hp['num_workers']

    save_dataset_folder = os.path.join('output','dataset', dataset_name)
    if not os.path.isdir(save_dataset_folder):
        os.mkdir(save_dataset_folder)
    save_dataset_path = os.path.join(save_dataset_folder, dataset_name + '.pkl')

    # apply training transforms and testing transforms separately
    tforms_train = Compose([Rescale(rescale_size),
                            RandomHorizontalFlip(0.5),
                            Blur(5, (0.5, 2.0)),
                            ToTensor()])
    tforms_test = Compose([Rescale(rescale_size),
                            ToTensor()])

    # create dataset objects
    ds_tform_train = SerratedTussockDataset(root_dir=root_dir,
                                            json_file=json_file,
                                            transforms=tforms_train)
    ds_tform_test = SerratedTussockDataset(root_dir=root_dir,
                                            json_file=json_file,
                                            transforms=tforms_test)

    # class definitions
    class_names = ["_background_", "serrated tussock"]

    # split into training, validation and testing - note: we're only after the indices here
    nimg = len(ds_tform_test)
    print('total number of images in the entire dataset: {}'.format(nimg))

    # currently, for Tussock_v0, we have 570 total images, so we split:
    # train: 500, val: 1 (no early stopping anyways), 69: (testing)
    tr = dataset_lengths[0]
    va = dataset_lengths[1]
    # te = nimg - tr - va
    te = dataset_lengths[2]
    ds_train, ds_val, ds_test = torch.utils.data.random_split(ds_tform_test,
                                                            [tr, va, te])

    # make sure to apply the training transforms only to the training dataset
    ds_train.dataset = ds_tform_train

    # setup dataloaders for efficient access to datasets
    dl_train = torch.utils.data.DataLoader(ds_train,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        collate_fn=collate_fn)

    dl_val = torch.utils.data.DataLoader(ds_val,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        collate_fn=collate_fn)

    dl_test = torch.utils.data.DataLoader(ds_test,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        collate_fn=collate_fn)

    # save datasets for later use (eg, training, etc)
    with open(save_dataset_path, 'wb') as f:
        pickle.dump(ds_tform_test, f)
        pickle.dump(ds_tform_train, f)
        pickle.dump(ds_train, f)
        pickle.dump(ds_val, f)
        pickle.dump(ds_test, f)
        pickle.dump(dl_test, f)
        pickle.dump(dl_train, f)
        pickle.dump(dl_val, f)
        pickle.dump(hp, f)

    print('datasets saved: {}'.format(save_dataset_path))

# -------------------------------------------------------------------- #
if __name__ == "__main__":

    # hyperparameters for dataset
    batch_size = 10
    num_workers = 10
    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 0.0001
    num_epochs = 100
    step_size = round(num_epochs / 2)
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
    hp['rescale_size'] = rescale_size

    # dataset folder/file settings
    dataset_name = 'Tussock_v0'
    root_dir = os.path.join(os.sep, 'home', 'dorian', 'Data', 'AOS_TussockDataset', 'Tussock_v0')
    json_file = os.path.join('Annotations', 'annotations_tussock_21032526_G507_combined.json')

    dataset_lengths = (500, 1, 69)
    split_dataset(root_dir,
                json_file,
                hp,
                dataset_name,
                dataset_lengths)

    import code
    code.interact(local=dict(globals(), **locals()))
