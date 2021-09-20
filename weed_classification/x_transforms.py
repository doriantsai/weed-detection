#! /usr/bin/env python

""" script to test image transform classes """

# create dataset
# for one instance of dataset, call get_item


import os
from deepweeds_dataset import DeepWeedsDataset, Rescale, RandomRotate, RandomAffine, RandomPixelIntensityScaling, Compose, ToTensor
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


img_dir = os.path.join('images')
lbl_file = os.path.join('labels', 'labels.csv')

lbls = pd.read_csv(lbl_file)

tform = Compose([RandomPixelIntensityScaling(scale_min=1.25, scale_max=1.25)])

dw = DeepWeedsDataset(lbl_file, img_dir, transform=tform)


sample = next(iter(dw))


img = tform(sample)
img = sample['image']
lbl = sample['label']
img_id = sample['image_id']
# img is a PIL image
img.show()

img = np.array(img)
print(img[0:10, 0:10, 0])

# original image:
img_name = lbls.iloc[img_id, 0]
img_o = Image.open(os.path.join(img_dir, img_name))
img_o.show()



img_o = np.array(img_o)
print(img_o[0:10, 0:10, 0])

import code
code.interact(local=dict(globals(), **locals()))

