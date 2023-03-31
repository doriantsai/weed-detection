#! /usr/bin/env/python3

""" model training script """

import os
import weed_detection.TrainMaskRCNN as TrainmaskRCNN

## define data folders, and locations of key configuration files

# root_dir of dataset
# root_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_clarkefield31'
root_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellanglo29'
# root_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellanglo32'

# code_base dir
code_dir = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection'

# annotation file, correesponding image directory and mask directory
annotation_data = {'annotation_file': os.path.join(root_dir, 'dataset.json'),
                    'image_dir': os.path.join(root_dir, 'annotated_images'),
                    'mask_dir': os.path.join(root_dir, 'masks')}


# training to validation ratio out of 1, remainder is for test
train_val_ratio = (0.8, 0.15)

# default output directory, where models and progress checkpoints are saved
output_dir = os.path.join(code_dir, 'model')

# default config file for class names and colours
classes_config = os.path.join(code_dir, 'config/classes.json')

# default image list text files for training, validation and testing:
imagelist_files = {'train_file': os.path.join(root_dir, 'metadata/train.txt'),
                    'val_file': os.path.join(root_dir, 'metadata/val.txt'),
                    'test_file': os.path.join(root_dir, 'metadata/test.txt')}

# default hyper parameter text file for training the model
hyper_parameters = os.path.join(code_dir, 'config/hyper_parameters.json')


##################################################################################

# initiate training object
TrainMask = TrainmaskRCNN.TrainMaskRCNN(annotation_data=annotation_data,
                                        train_val_ratio=train_val_ratio,
                                        imagelist_files=imagelist_files,
                                        output_dir=output_dir,
                                        classes_config_file=classes_config,
                                        hyper_param_file=hyper_parameters)
# call training pipeline
TrainMask.train_pipeline()

print('---------------------------------------------------------------------')



