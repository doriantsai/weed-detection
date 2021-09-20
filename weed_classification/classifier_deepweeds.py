#! /usr/bin/env python

# manyweeds.py
# script to show how many weed images are necessary

# conversion to script from early notebook exploration

from __future__ import print_function, division
from math import degrees
import os
import sys
import pandas as pd
from skimage import io, transform

import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pickle

import code

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

# use tensorboard to see training in progress
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from deepweeds_dataset import DeepWeedsDataset, Rescale, RandomAffine, RandomColorJitter, RandomPixelIntensityScaling
from deepweeds_dataset import RandomHorizontalFlip, RandomRotate, RandomResizedCrop, ToTensor, Compose
from deepweeds_dataset import CLASSES, CLASS_NAMES


np.random.seed(42)
torch.manual_seed(42)





############# functions #################


def show_image(image, label):
    """ show image with label """
    if not isinstance(image, np.ndarray):
        raise TypeError(image, 'invalid image input type (want np.ndarray)')
    if not isinstance(label, np.integer):
        raise TypeError(label, 'invalid label input type (want int)')
    # if not isinstance(weed_name, str):
    #     raise TypeError(weed_name, 'invalid type (want str)')

    # show image
    plt.imshow(image)
    xy = (5, image.shape[1]/20)  # annotation offset from top-left corner
    ann = str(label)  # + ': ' + weed_name
    plt.annotate(ann, xy, color=(1, 0, 0))
    plt.pause(0.001)


# helper function to show a batch
def show_weeds_batch(sample_batched):
    """
    Show image with weed name for batch of samples
    """
    images_batch, weeds_batch = \
        sample_batched['image'], sample_batched['weed']
    bs = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    # print(weed_names_batch)
    # annotate with weed names:
    for i in range(bs):
        x = 0 + i * im_size + (i + 1) * grid_border_size
        y = 0 + im_size / 10
        plt.annotate(weeds_batch[i], xy=(x, y), color=(1, 0, 0))

    plt.title('Batch from dataloader')
    # plt.show()


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)

    preds = np.squeeze(preds_tensor.cpu().numpy())

    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    nimgs = len(images)
    if nimgs > 4:
        nimgwidth = 4
    else:
        nimgwidth = nimgs
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 4))
    for idx in np.arange(nimgwidth):
        ax = fig.add_subplot(1, nimgwidth, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            CLASS_NAMES[preds[idx]],
            probs[idx] * 100.0,
            CLASS_NAMES[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


# def show_image_grid(images, labels=None, predictions=None):
#     """ show grid of images, labels and predictions """
#     # images = tensors
#     bs = len(images)
#     imsize = images.size(2)

#     # plot grid of images
#     grid_border_size = 2
#     grid = utils.make_grid(images)
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))

#     # plot labels (groundtruth)
#     for i in range(bs):
#         x = 0 + i * imsize + (i + 1) * grid_border_size
#         y = 0 + imsize / 10
#         if labels:
#             plt.annotate(CLASS_DICT[labels[i]], xy=(x, y), color=(1, 0, 0))
#         if predictions:
#             plt.annotate(CLASS_DICT[predictions[i]], xy = (x, y * 2), color=(0, 1, 0))

#     plt.title('batch from dataloader')


def train_model(model,
                train_dl,
                valid_dl,
                loss_fn,
                optimizer,
                acc_fn,
                num_epochs=1,
                save_path='manyweeds_train.pth',
                train_size=None,
                val_size=None,
                expname=None,
                acc_step_size=50):
    """
    train the model using the training dataloaders (train_dl), check accuracy of model using validation dataloader (valid_dl)

    """
    start = time.time()

    if train_size is None:
        # a very rough approximation, but should hit the right ballpark
        train_size = len(train_dl) * train_dl.batch_size
    if val_size is None:
        val_size = len(valid_dl) * valid_dl.batch_size

    train_loss, valid_loss = [], []

    # use gpu if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.to(device)

    # report progress to Tensorboard (use 'tensorboard --logdir=runs' in terminal)
    if expname is None:
        now = str(datetime.datetime.now())
        expname = now[0:10] + '_' + now[11:13] + '-' + now[14:16] + '-Run'
    writer = SummaryWriter(os.path.join('runs', expname))

    # best_acc = 0.0

    print('begin training {} epochs'.format(num_epochs))

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # first we are in training phase, then we are in validation phase
        # for phase in ['train', 'validate']:
        for phase in ['train']:
            # not sure why this doesn't work!
            # if (not (epoch % 50 == 49)) and (phase == 'validate'):
            #     # only do the validation every 50 epochs to save on compute
            #     break

            if phase == 'train':
                # print('--Training Phase--')
                model.train(True)
                dataloader = train_dl
            else:
                # print('--Validation Phase--')
                # model.train(False)
                model.eval()
                dataloader = valid_dl
            running_loss = 0.0
            running_acc = 0.0

            for sample_batch in dataloader:
                # note: output from dataloader is all converted to tensors, I believe
                imgs, lbls = sample_batch['image'], sample_batch['label']


                # We move our tensor to the GPU if available
                if torch.cuda.is_available():
                    imgs = imgs.to(device)
                    lbls = lbls.to(device)

                # the backward pass frees the graph memory, so there is no
                # need for torch.no_grad in this training pass

                if phase == 'train':
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = loss_fn(outputs, lbls)
                    loss.backward()
                    optimizer.step()
                else:
                    # in validation phase
                    with torch.no_grad():
                        outputs = model(imgs)
                        loss = loss_fn(outputs, lbls.long())

                # stats
                # acc = acc_fn(outputs, lbls)

                # running_acc += acc * dataloader.batch_size
                running_loss += loss * dataloader.batch_size

                # if step % 10 == 0:
                #     print('Currenty step: {}  Loss: {}  Acc: {}'.format(step, loss, acc))

            epoch_loss = running_loss / len(dataloader.dataset)
            # epoch_acc = running_acc / len(dataloader.dataset)

            # print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            nImgPerClass = round((train_size + val_size) / len(CLASS_NAMES))

            writer.add_scalar('Size' + str(nImgPerClass) +
                              '/Training_Loss', epoch_loss, epoch + 1)
            # writer.add_scalar('Size' + str(train_size + val_size) + '/Accuracy', epoch_acc, epoch + 1)
            # writer.add_graph(model,imgs)

            # writer.add_figure('predictions vs actuals',
            #                   plot_classes_preds(model, imgs, lbls))

            # train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)
            # end of training/validation phases, to the next epoch!

        if (epoch % acc_step_size) == (acc_step_size - 1):
            # run an accuracy/evaluate model every 50 epochs:
            model.eval()
            dataloader = valid_dl
            running_loss_val = 0.0
            running_acc = 0.0

            for step, sample_batch in enumerate(dataloader):
                # note: output from dataloader is all converted to tensors, I believe
                imgs, lbls = sample_batch['image'].float(
                ), sample_batch['label']
                # We move our tensor to the GPU if available
                if torch.cuda.is_available():
                    imgs = imgs.to(device)
                    lbls = lbls.to(device)

                # in validation phase
                with torch.no_grad():
                    outputs = model(imgs)
                    loss = loss_fn(outputs, lbls.long())

                # stats
                acc = acc_fn(outputs, lbls)
                running_acc += acc * dataloader.batch_size
                running_loss_val += loss * dataloader.batch_size

            epoch_acc = running_acc / len(dataloader.dataset)
            epoch_loss = running_loss_val / len(dataloader.dataset)
            nImgPerClass = round((train_size + val_size) / len(CLASS_NAMES))
            writer.add_scalar('Size' + str(nImgPerClass) +
                              '/Val_Accuracy', epoch_acc, epoch + 1)
            writer.add_scalar('Size' + str(nImgPerClass) +
                              '/Val_Loss', epoch_loss, epoch + 1)

            sample_batch = next(iter(dataloader))

            # import code
            # code.interact(local=dict(globals(), **locals()))

            imgs, lbls = sample_batch['image'].float(), sample_batch['label']
            if torch.cuda.is_available():
                imgs = imgs.to(device)
                lbls = lbls.to(device)
            writer.add_figure('predictions vs actuals',
                              plot_classes_preds(model, imgs, lbls))

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    writer.close()

    print('saving model')
    torch.save(model.state_dict(), save_path)

    return train_loss, valid_loss


def acc_metric(outputs, labels):
    # use gpu if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)
    if torch.cuda.is_available():
        labels.to(device)
    _, predicted = torch.max(outputs, 1)

    return (predicted == labels).float().mean()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    # main code
    # init classes as objects
    # setup dataset, dataloaders
    # run training + validation
    # save progress on tensorboard
    # save results
    # test on whole dataset
    # print final things

    # folder settings
    model_folder = './models'
    images_folder = './images'
    labels_folder = './labels'
    # save_path = './saved_model/testtraining/deepweeds_resnet50_train0.pth'

    # read labels
    # labels_file = os.path.join(labels_folder, 'nonnegative_labels.csv')
    lbls_file = 'development_labels_trim.csv'
    labels_file = os.path.join(labels_folder, lbls_file)

    # labels = pd.read_csv(labels_file)

    # classes
    # see global?

    # hyperparameters -->now taken from the DeepWeeds Paper
    num_epochs = 200
    learning_rate = 0.0001 # LR halved if val loss did not decrease after 16 epochs
    momentum = 0.9
    batch_size = 10 # TODO use batch size 32
    shuffle = True
    num_workers = 10
    expsuffix = 'tbplots'
    acc_step_size = 5
    # TODO early stopping if val loss did not decrease after 32 epochs

    # assuming an even class distribution, if we want training size of X/class, then we need
    # X*8 images
    print('classes: {}'.format(len(CLASSES)))
    train_test_split = 0.9

    # max images = 8403 * 0.8 / 8 = 840.3
    # ntrain]  # last number should be ntrain
    train_sizes = [500 * len(CLASSES)]
    print('train size = {}'.format(train_sizes))

    ntrain_times = 1

    folder_num = 1  # folder number for training - sub-folder ID

    # TODO could have also used the random_split()
    # train_label_file = os.path.join(labels_folder, 'train_subset0.csv')
    # val_label_file = os.path.join(labels_folder, 'val_subset0.csv')  # perhaps join these?
    # test_label_file = os.path.join(labels_folder, 'test_subset0.csv')

    # tforms = transforms.Compose([
    #                             Rescale(256),
    #                             RandomCrop(224),
    #                             ToTensor()
    #                             ])

    # TODO actually, I think I have to write classes for each, because they return sample, rather than just image...
    tforms = Compose([
        Rescale(256),
        RandomRotate(360),
        RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
        RandomColorJitter(brightness=(0, 0.1), hue=(-0.01, 0.01)), # TODO unsure about these values
        RandomPixelIntensityScaling(),
        RandomAffine(degrees=5, translate=(0.05, 0.05)),
        RandomHorizontalFlip(prob=0.5),
        ToTensor()
    ])
    # full dataset
    full_dataset = DeepWeedsDataset(csv_file=labels_file,
                                    root_dir=images_folder,
                                    transform=tforms)
    nimg = len(full_dataset)
    print('full_dataset length =', nimg)

    # edit the dataset - remove the negative class:
    # go through the csv file, make a new csv file with no labels from the negative class
    # save the corresponding images
    # copy over new set of images to a new folder?

    # use random_split to break up train/val/test sets
    # then further use random_split to break train into smaller and smaller sets

    # first, take the largest training size, and split it into training and testing:
    # choose ratio of 80/20 % for training/testing

    ntrain = round(nimg * train_test_split)

    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [ntrain, nimg - ntrain])

    # now we split up the original train_dataset into subdatasets
    # each needs their training and validation datasets
    train_val_split = 0.7
    train_dataset_list = []
    val_dataset_list = []
    train_dataloader_list = []
    val_dataloader_list = []
    for i, ts in enumerate(train_sizes):
        # first, we need to split the original train_dataset into train_sizes[i]
        tsd = torch.utils.data.random_split(train_dataset,
                                            [train_sizes[i], len(train_dataset) - train_sizes[i]])[0]

        nt = round(train_sizes[i] * train_val_split)
        nv = train_sizes[i] - nt
        print(len(train_dataset))
        print(train_sizes[i])
        print(nt)
        print(nv)
        td, vd = torch.utils.data.random_split(tsd, [nt, nv])

        train_dataset_list.append(td)
        val_dataset_list.append(vd)

        # make dataloaders for each dataset
        tdl = DataLoader(td,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=num_workers)
        vdl = DataLoader(vd,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=num_workers)

        train_dataloader_list.append(tdl)
        val_dataloader_list.append(vdl)

    # testing
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)
    print('test_dataset length =', len(test_dataset))
    print(f'validation dataset length = {len(vd)}')
    print(f'traininh dataset length = {len(td)}')
    # save datasets and dataloader objects for future use:
    save_dataset_objects = os.path.join('dataset_objects', lbls_file[:-4])
    os.makedirs(save_dataset_objects, exist_ok=True)
    save_dataset_path = os.path.join(
        save_dataset_objects, lbls_file[:-4] + '.pkl')
    with open(save_dataset_path, 'wb') as f:
        # TODO should save the entire dataset as well
        # pickle.dump(full_dataset, f)
        pickle.dump(td, f)
        pickle.dump(vd, f)
        pickle.dump(test_dataset, f)
        pickle.dump(tdl, f)
        pickle.dump(vdl, f)
        pickle.dump(test_loader, f)

    # code to iterate through dataset samples
    # for i in range(len(train_dataset)):
    #     sample = train_dataset[i]
    #     img, lbl = sample['image'], sample['label']
    #     print(i, img.shape, lbl)
    #     print(type(img))
    #     print(type(lbl))
    #     print()
    #     if i == 3:
    #         break

    # code to iterate through dataset using dataloader (converted to tensors)
    # print('test label')
    # for epoch in range(num_epochs):
    #     print('epoch {}'.format(epoch))
    #     running_loss = 0.0
    #     for i, sample_batch in enumerate(train_loader):
    #         imgs, lbls = sample_batch['image'].float(), sample_batch['label']

    #         print('  ', i)
    #         print('  imgs size =', imgs.size())
    #         # print('  imgs type =', type(imgs))
    #         print('  lbls =', lbls)
    #         # print('  lbls type = ', type(lbls))

    #         if i == 2:
    #             break

    # define model
    # try replacing the last layer by first removing the last layer
    # short_model= nn.Sequential(*(list(original_model.children())[:-1]))
    # print(short_model)
    # classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(n_inputs, 8))]))
    train_loss_sizes = []
    val_loss_sizes = []
    save_paths = []
    model_acc_sizes = []

    for i, training_step_size in enumerate(train_sizes):
        train_loss_times = []
        val_loss_times = []
        model_acc_times = []

        for j in range(ntrain_times):

            # define model (resnet 50, random initialised weights)
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(in_features=2048, out_features=len(CLASSES), bias=True)

            # select loss function and optimizer
            # TODO consider adding weights to classes (eg Negative class)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(),
                                  lr=learning_rate, momentum=momentum)

            # generate save_path/name for each one
            # save_path = './saved_model/testtraining/deepweeds_resnet50_train0.pth'
            save_folder = os.path.join(
                'saved_model', 'development_training' + str(folder_num))
            # if not os.path.isdir(save_folder):
            #     os.mkdir(save_folder)
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, 'dw_r50_s' + str(
                round(training_step_size/len(CLASSES))) + '_i' + str(j) + '.pth')

            print(save_path)

            # generate run name (for tensorboard)
            now = str(datetime.datetime.now())
            expname = now[0:10] + '_' + now[11:13] + \
                '-' + now[14:16] + '-' + expsuffix
            # NOTE: model saved at end of train_model using save_path
            train_loss_i, val_loss_i = train_model(model,
                                                   train_dataloader_list[i],
                                                   val_dataloader_list[i],
                                                   criterion,
                                                   optimizer,
                                                   acc_metric,
                                                   train_size=len(
                                                       train_dataset_list[i]),
                                                   val_size=len(
                                                       val_dataset_list[i]),
                                                   num_epochs=num_epochs,
                                                   save_path=save_path,
                                                   expname=expname,
                                                   acc_step_size=acc_step_size)

            # evaluate model using the test set:
            model.eval()

            dataiter = next(iter(test_loader))
            images, labels = dataiter['image'], dataiter['label']
            nimg = len(images)
            print(images.size())
            print(labels)
            print('Groundtruth: ', ' '.join('%5s' %
                  CLASSES[labels[j]] for j in range(nimg)))

            outputs = model(images.float().cuda())
            _, predicted = torch.max(outputs, 1)
            print('Predicted: ', ' '.join('%5s' %
                  CLASSES[predicted[j]] for j in range(nimg)))

            # plot to tensorboard
            # create grid of images
            # img_grid = torchvision.utils.make_grid(images)
            # write to tensorboard
            # show_image_grid(images, labels, predicted)
            # test entire network:
            correct = 0
            total = 0

            # use gpu if available
            device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
            # print(device)
            # if torch.cuda.is_available():
            #     model.to(device)
            # assume that model.to(device) already on the GPU from training

            with torch.no_grad():
                for data in test_loader:
                    images, labels = data['image'].float(), data['label']
                    if torch.cuda.is_available():
                        images = images.to(device)
                        labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            model_acc = 100 * correct / total
            print('Accuracy of network on {} test images: {}'.format(
                total, model_acc))

            # save the training and loss numbers
            train_loss_times.append(train_loss_i)
            val_loss_times.append(val_loss_i)
            save_paths.append(save_path)
            model_acc_times.append(model_acc)

            # del model

            # import code
            # code.interact(local=dict(globals(), **locals()))

        train_loss_sizes.append(train_loss_times)
        val_loss_sizes.append(val_loss_times)
        model_acc_sizes.append(model_acc_times)

    # save train_loss_sizes and val_loss_sizes just in case:
    save_tv_path = os.path.join('.', 'saved_model', 'training' + str(
        folder_num), 'dw_train_val_losses' + str(folder_num) + '.pkl')
    with open(save_tv_path, 'wb') as f:
        pickle.dump(train_loss_sizes, f)
        pickle.dump(val_loss_sizes, f)
        pickle.dump(model_acc_sizes, f)

    # load model, assuming it has been saved in save_path
    # TODO put an "if it exists statement"
    # TODO load from save_paths[i]
    # model.load_state_dict(torch.load(save_path))

    # print out size of all variables in bytes:
    # we are looking for ones outrageously large
    # local_vars = list(locals().items())
    # for var, obj in local_vars:
    #     print(var, sys.getsizeof(obj))

    import code
    code.interact(local=dict(globals(), **locals()))
