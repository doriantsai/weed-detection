#! /usr/bin/env python

# manyweeds.py
# script to show how many weed images are necessary

# conversion to script from early notebook exploration

import os

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pickle

import code

from torch.utils.data import DataLoader
from torchvision import utils, models

# use tensorboard to see training in progress
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from deepweeds_dataset import DeepWeedsDataset, Rescale, RandomAffine, RandomColorJitter, RandomPixelIntensityScaling
from deepweeds_dataset import RandomHorizontalFlip, RandomRotate, RandomResizedCrop, ToTensor, Compose
from deepweeds_dataset import CLASSES, CLASS_NAMES


np.random.seed(42)
torch.manual_seed(42)



############# functions #################


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

    # how long/size of dataset
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
    # TODO snapshots/checkpoints

    # =========================================================================
    # set folders/files
    # =========================================================================
    # folder settings
    images_folder = 'images'
    labels_folder = 'labels'

    # read labels
    lbls_file = 'development_labels_trim.csv'
    lbls_path = os.path.join(labels_folder, lbls_file)

    # create unique name folder folder/model based on time of running this code
    now = str(datetime.datetime.now())
    nowstr = now[0:10] + '-' + now[11:13] + '-' + now[14:16]
    model_name = 'deepweeds_r50_' + nowstr
    model_folder = os.path.join('output', model_name)
    os.makedirs(model_folder, exist_ok=True)

    # =========================================================================
    # set hyperparameters
    # =========================================================================
    # roughly taken from the DeepWeeds Paper
    num_epochs = 500
    learning_rate = 0.001 # originally 0.001, LR 1/2 if val loss did not decrease after 16 epochs
    momentum = 0.9 # not specified
    batch_size = 150 # 32 in paper can probably get away with more, 90 = 12/24 GB
    shuffle = True
    num_workers = 10
    expsuffix = 'dev'
    acc_step_size = 5
    # TODO early stopping if val loss did not decrease after 32 epochs

    # =========================================================================
    # create datasets
    # =========================================================================

    # transforms
    tforms_train = Compose([
        Rescale(256),
        RandomRotate(360),
        RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
        RandomColorJitter(brightness=(0, 0.1), hue=(-0.01, 0.01)), # TODO unsure about these values
        RandomPixelIntensityScaling(),
        RandomAffine(degrees=5, translate=(0.05, 0.05)),
        RandomHorizontalFlip(prob=0.5),
        ToTensor()
    ])

    full_data = DeepWeedsDataset(lbls_path, images_folder, tforms_train)
    nimg = len(full_data)
    print('full dataset length =', nimg)

    # set train/val/test split ratio
    data_split = [0.6, 0.2, 0.2] # TODO ensure sums to 1

    nimg_train = int(round(nimg * data_split[0]))
    nimg_val = int(round(nimg * data_split[1]))
    nimg_test = nimg - nimg_train - nimg_val

    print('n_train {}'.format(nimg_train))
    print('n_test {}'.format(nimg_val))
    print('n_val {}'.format(nimg_test))

    ds_train, ds_val, ds_test = torch.utils.data.random_split(full_data, [nimg_train, nimg_val, nimg_test])

    # csv files that dictate train/test/val
    # lbls_train_file = lbls_file[:-4] + '_train.csv'
    # lbls_val_file = lbls_file[:-4] + '_val.csv'
    # lbls_test_file = lbls_file[:-4] + '_test.csv'

    # since we don't want to randomise the test set, just rescale to 224 and make a tensor
    tforms_test = Compose([
        Rescale(224),
        ToTensor()
    ])
    ds_test.dataset.set_transform(tforms_test)

    # make dataloaders for each dataset
    dl_train = DataLoader(ds_train,
                     batch_size=batch_size,
                     shuffle=True,
                     num_workers=num_workers)
    dl_val = DataLoader(ds_val,
                     batch_size=batch_size,
                     shuffle=True,
                     num_workers=num_workers)
    dl_test = DataLoader(ds_test,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=num_workers)

    # save datasets and dataloader objects for future use:
    save_data_dir = os.path.join(model_folder, lbls_file[:-4])
    os.makedirs(save_data_dir, exist_ok=True)
    save_data_path = os.path.join(save_data_dir, lbls_file[:-4] + '.pkl')
    with open(save_data_path, 'wb') as f:
        pickle.dump(full_data, f)
        pickle.dump(ds_train, f)
        pickle.dump(ds_val, f)
        pickle.dump(ds_test, f)
        pickle.dump(dl_train, f)
        pickle.dump(dl_val, f)
        pickle.dump(dl_test, f)

    # =========================================================================
    # create datasets
    # =========================================================================

    # define model
    train_loss_sizes = []
    val_loss_sizes = []
    save_paths = []
    model_acc_sizes = []

    train_loss_times = []
    val_loss_times = []
    model_acc_times = []

    # define model (resnet 50, random initialised weights)
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(in_features=2048, out_features=len(CLASSES), bias=True)

    # model fron deep weeds, which was converted from keras to pytorch
    # keras_model_path =
    # model.load_state_dict()

    # select loss function and optimizer
    # TODO consider adding weights to classes (eg Negative class?)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # generate save_path/name for each one
    save_path = os.path.join(model_folder, model_name + '.pth')

    # generate run name (for tensorboard)
    expname = nowstr + '-' + expsuffix
    # NOTE: model saved at end of train_model using save_path
    train_loss, val_loss = train_model(model,
                                       dl_train,
                                       dl_val,
                                       criterion,
                                       optimizer,
                                       acc_metric,
                                       train_size=len(ds_train),
                                       val_size=len(ds_val),
                                       num_epochs=num_epochs,
                                       save_path=save_path,
                                       expname=expname,
                                       acc_step_size=acc_step_size)

    # evaluate model using the test set:
    model.eval()

    dataiter = next(iter(dl_test))
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # assume that model.to(device) already on the GPU from training

    with torch.no_grad():
        for data in dl_test:
            images, labels = data['image'].float(), data['label']
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model_acc = 100 * correct / total
    print('Accuracy of network on {} test images: {}'.format(total, model_acc))

     # save train_loss_sizes and val_loss_sizes just in case:
    save_tv_path = os.path.join(model_folder, model_name + '_losses.pkl')
    with open(save_tv_path, 'wb') as f:
        pickle.dump(train_loss_sizes, f)
        pickle.dump(val_loss_sizes, f)
        pickle.dump(model_acc_sizes, f)

    import code
    code.interact(local=dict(globals(), **locals()))
