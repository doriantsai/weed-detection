#! /usr/bin/env python

"""
a class to group collection of model inference and plotting functions
"""

# import pkl file of dataset/dataloader
# iterate through dataloader
# plot image with name and confidence on each image
# save image
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torchf
import pickle as pkl
import cv2 as cv

from torchvision import utils, models
from torch.utils.data import DataLoader
from deepweeds_dataset import DeepWeedsDataset, CLASSES, CLASS_NAMES, CLASS_COLORS, CLASS_COLOR_ARRAY
from deepweeds_dataset import Compose, ToTensor, RandomResizedCrop


class DeepweedsModel(object):

    def __init__(self,
                 model_name,
                 model_path,
                 model=None,
                 data_path=None,
                 img_dir=None,
                 lbl_file=None,
                 device=None):

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._device = device

        self._model_name = model_name
        self._model_path = model_path


        # load model
        if model is None and model_path is not None:
            self.load_model()
        else:
            self._model = model

        # TODO load the model?
        self._data_path = data_path
        self._img_dir = img_dir
        self._lbl_file = lbl_file




    # getters
    def get_model_name(self):
        return self._model_name

    def get_model_path(self):
        return self._model_path

    def get_data_path(self):
        return self._data_path

    def get_img_dir(self):
        return self._img_dir

    def get_lbl_file(self):
        return self._lbl_file

    def get_model(self):
        return self._model

    # setters
    def set_model_name(self, model_name):
        # todo make sure is string
        self._model_name = model_name

    def set_model_path(self, model_path):
        self._model_path = os.path.join(model_path)
        # todo automatically load the model?

    def set_model(self, model):
        self._model = model

    def set_data_path(self, data_path):
        self._data_path = data_path

    def set_img_dir(self, img_dir):
        self._img_dir = img_dir

    def set_lbl_file(self, lbl_file):
        self._lbl_file = lbl_file


    def load_data(self, data_path=None):
        """ loads data path"""

        if data_path is None:
            data_path = self.get_data_path()

        if data_path is None: # still
            print('Error: data_path is None')
            return False
        else:
            if os.path.isfile(data_path):
                with open(self._data_path, 'rb') as f:
                    # dw_data_dict = pkl.load(f)
                    full_data = pkl.load(f)
                    ds_train = pkl.load(f)
                    ds_val = pkl.load(f)
                    ds_test = pkl.load(f)
                    dl_train = pkl.load(f)
                    dl_val = pkl.load(f)
                    dl_test = pkl.load(f)

                    dw_data = {'full_data': full_data,
                                'ds_train': ds_train,
                                'ds_val': ds_val,
                                'ds_test': ds_test,
                                'dl_train': dl_train,
                                'dl_val': dl_val,
                                'dl_test': dl_test}
                return dw_data

            else:
                print('Error: data_path is not a file')
                print(f'data_path = {data_path}')
                return False


    def build_model(self, in_features=2048, out_features=len(CLASSES), bias=True):
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        # self._model = model
        self.set_model(model)


    def load_model(self, model_path=None, model_name='resnet50'):
        """ load model path """
        if model_path is None and self._model_path is None:
            print('Error: no valid model path')
            return False
        elif model_path is None:
            model_path = self._model_path
        else:
            model_path = model_path

        self.build_model()
        self._model.load_state_dict(torch.load(model_path))
        print('loaded model: {}'.format(model_path))
        self._model.to(self._device)
        self._model_name = model_name


    def show_image(self,
                   img,
                   lbl=None,
                   pred=None,
                   pred_conf=None,
                   img_name=None,
                   save_dir=None,
                   save_img=False):
        """ show image with label and confidence """
        # TODO if pred, then show pred
        # TODO saving image, name and directory
        # TODO why flip-flop between matplotlib and opencv? chose one.

        font_scale = 0.5 # 1.0/224.0
        font_thick = 2
        pred_color_right = [0, 200, 0] # RGB
        pred_color_wrong = [255, 0, 0] # RGB
        pred_color_neutral = [200, 200, 200]

        # if not isinstance(img, np.ndarray):
        #     raise TypeError(img, 'invalid image input type (want np.ndarray)')
        # if not isinstance(lbl, np.integer):
        #     raise TypeError(lbl, 'invalid label input type (want int)')
        # if not isinstance(weed_name, str):
        #     raise TypeError(weed_name, 'invalid type (want str)')

        # show image
        # plt.imshow(img)
        # xy = (5, img.shape[1]/20)  # annotation offset from top-left corner
        # # also add confidence score
        # ann = str(lbl)  # + ': ' + weed_name
        # plt.annotate(ann, xy, color=(1, 0, 0))
        # plt.pause(0.001)

        # assuming input image is a tensor
        img_out = img.cpu().numpy()
        # BGR as opposed to RGB
        img_out = np.transpose(img_out, (1,2,0))
        img_out = cv.normalize(img_out, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        # annotate groundtruth label
        if lbl is not None:
            cv.putText(img_out,
                       'label: {}'.format(CLASS_NAMES[lbl]),
                       (int(5), int(img.shape[1]/15) + 5),
                       fontFace=cv.FONT_HERSHEY_COMPLEX,
                       fontScale=font_scale,
                       color=CLASS_COLOR_ARRAY[lbl] * 255,
                       thickness=font_thick)

        if (pred is not None) and (pred_conf is not None):
            pred_sc = format(pred_conf * 100.0, '.0f') # prediction score %, no decimals
            if lbl is not None:
                if pred == lbl:
                    text_color = pred_color_right
                else:
                    text_color = pred_color_wrong
            else:
                text_color = pred_color_neutral

            cv.putText(img_out,
                       'pred: {}, {}'.format(CLASS_NAMES[pred], pred_sc),
                       (int(5), int(img.shape[1]/15 + 30)),
                       fontFace=cv.FONT_HERSHEY_COMPLEX,
                       fontScale=font_scale,
                       color=text_color,
                       thickness=font_thick)

        if save_img:
            if save_dir is None:
                save_dir = os.path.join('output')
            save_img_name = os.path.join(save_dir, os.path.basename(img_name)[:-4] + '_pred.png')
            print(save_img_name)
            img_out_bgr = cv.cvtColor(img_out, cv.COLOR_RGB2BGR)
            cv.imwrite(save_img_name, img_out_bgr)

        return img_out


    def infer_images(self, imgs):
        """ generates predictions and corresponding confidence scores from
        trained network and list of images"""
        # TODO doesn't work yet

        self._model.eval()
        with torch.no_grad():
            print('debug at infer_image')
            import code
            code.interact(local=dict(globals(), **locals()))
            outs = self._model(imgs)

        # convert output confidences to predicted classes
        _, preds = torch.max(outs, 1)
        preds = np.squeeze(preds.cpu().numpy())

        # classes
        pred_classes = [torchf.softmax(el, dim=0)[i].item() for i, el in zip(preds, outs)]

        return preds, pred_classes


    def infer_images_batch(self, imgs_batch):
        """ generates predictions and corresponding confidence scores from
        trained network and batch of images """

        self._model.eval()

        with torch.no_grad():
            imgs_batch = imgs_batch.to(self._device)
            self._model.to(self._device)

            outs = self._model(imgs_batch)

            # convert output confidences to predicted classes
            _, preds = torch.max(outs, 1)
            preds = np.squeeze(preds.cpu().numpy())

            # classes
            pred_confidences = [torchf.softmax(el, dim=0)[i].item() for i, el in zip(preds, outs)]
        return preds, pred_confidences


    def infer_data(self, dataloader, save_dir=None):
        """ method to take a dataset, output images/predictions """

        print('number of batches of images to infer: {}'.format(len(dataloader)))
        predictions = []
        for data in dataloader:
            imgs, lbls, ids = data['image'], data['label'], data['image_id']
            img_names = [dataloader.dataset.get_image_name(i.item()) for i in ids]

            # get predictions on image
            pred, pred_conf = self.infer_images_batch(imgs)

            # annotate image with predictions
            if save_dir is None:
                save_img = False
            else:
                save_img = True

            for i, img in enumerate(imgs):
                _ = self.show_image(img,
                                    lbls[i],
                                    pred[i],
                                    pred_conf[i],
                                    save_dir=save_dir,
                                    save_img=save_img,
                                    img_name=img_names[i])

            predictions.append(pred)
        return predictions


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    img_dir = 'images'
    lbl_dir = './labels'

    # main code
    model_name = 'deepweeds_r50_2021-09-22-14-30'
    # model_name = 'deepweeds_r50_2021-10-05-21-19'
    model_path = os.path.join('output',
                              model_name,
                              model_name + '.pth')

    # TODO need to bring in deployment csv file!! see prcurve.py, line 63
    lbls_file = 'deployment_labels.csv'

    lbls_file = os.path.join(lbl_dir, lbls_file)
    tforms = Compose([
        RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
        ToTensor()
    ])
    full_data = DeepWeedsDataset(lbls_file, img_dir, tforms)
    batch_size = 32
    num_workers = 10
    dataloader = DataLoader(full_data,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=num_workers)

    # data_path = os.path.join(os.path.dirname(model_path),
    #                         'development_labels_trim',
    #                         'development_labels_trim.pkl')


    # init object
    DW = DeepweedsModel(model_name=model_name,
                        model_path=model_path,
                        img_dir=img_dir)



    # test loading data
    # dw_data = DW.load_data()
    print(os.path.dirname(model_path))
    save_img_dir = os.path.join(os.path.dirname(model_path), 'deployment')
    os.makedirs(save_img_dir, exist_ok=True)
    pred = DW.infer_data(dataloader, save_dir=save_img_dir)


    import code
    code.interact(local=dict(globals(), **locals()))


