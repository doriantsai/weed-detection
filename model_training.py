#! /usr/bin/env/python3

""" model training script """

import weed_detection.TrainMaskRCNN as TrainmaskRCNN

## define data folders, and locations of key configuration files

# annotation file, correesponding image directory and mask directory
annotation_data = {'annotation_file': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_clarkfield31/dataset.json',
                    'image_dir': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_clarkfield31/annotated_images',
                    'mask_dir': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_clarkfield31/masks'}

# training to validation ratio out of 1, remainder is for test
train_val_ratio = (0.8, 0.15)

# default output directory, where models and progress checkpoints are saved
output_dir = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/model'


# default config file for class names and colours
classes_config = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/config/classes.json'

# default image list text files for training, validation and testing:
imagelist_files = {'train_file': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_clarkfield31/metadata/train.txt',
                    'val_file': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_clarkfield31/metadata/val.txt',
                    'test_file': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_clarkfield31/metadata/test.txt'}

# default hyper parameter text file for training the model
hyper_parameters = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/config/hyper_parameters.json'


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



