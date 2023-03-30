# QUT Weed Detection Package
A deep learning approach to pastoral weed detection. 


## Overview
A deep learning approach to weed species detection for smart farm pasture management developed by the Queensland University of Technology (QUT), in collaboration with Agent Oriented Systems (AOS), Department of Primary Industries (DPI) and the University of New England (UNE). The robotic system used for data collection was developed by AOS. Data collection were performed and polygon annotations were provided primarily by DPI and AOS. Weed detection models were then developed by both QUT and UNE. This repository is focused on providing a neural network training pipeline for weed detection using MaskRCNN.


## Data
Data is obtained from the DPI's Weed Reference Library, available at https://www.agkelpie.com/. Access is currently limited for potential commercial development reasons, but a selected amount will be publicly available for research purposes. 


## File and Folder Descriptions
* **weed_detection** - QUT's weed detector for multi-class weed detection
* **config** - configuration files for the virtual environment
* **model** - model-specific files (weights, species name for the corresponding model, etc)
* **examples** - a minimum working example for model inference with a sample image, designed to work with the Agkelpie weed_camera_interface code available at https://github.com/AgKelpie/weed_camera_interface
- *make_conda_agkelpie.sh* - the script that automatically creates the agkelpie conda virtual environment
- *setup.py* - along with __init__.py in weed_detection, required to install ``weed_detection`` as a python package


## Installation & Dependencies
- The following code was run in Ubuntu 20.04 LTS using Python 3
- Install conda https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html, although I recommend miniforge/mambaforge https://github.com/conda-forge/miniforge, which is a much more minimal and faster installer for conda packages. These installers will create a ''base'' environment that contains the package managers for conda.
- `conda install mamba -c conda-forge`
- `git clone git@github.com:AgKelpie/weed_detector_qut.git`
- `cd` to the weed_detector_qut folder
- Create conda environment automatically by running the script "make_conda_agkelpie.sh" file
    - make .sh file executable via `chmod +x make_conda_agkelpie.sh`
    - run .sh file via `./make_conda_agkelpie.sh`
    - Note: you can read the dependencies in the `agkelpie.yml` file
    - Experimental: if you want to go the `pip` route, specific requirements are given in `requirements.txt` file, generated automatically via `pipreqs` (`pip install pipreqs` then `pipreqs` in the repo home directory). You gain simplicity, but lose containerisation.
- this should automatically make a virtual environment called "agkelpie"
- to activate new environment: `conda activate agkelpie`
- to deactivate the new environment: `conda deactivate`


## WandB
- Weights & Biases is used to track and visualise the model training progress
- Setup an account https://docs.wandb.ai/quickstart
- Adjust the user details in TrainMaskRCNN.py `WANDB_PROJECT_NAME_DEFAULT` and 
`WANDB_USER_DEFAULT`


## Model Training
- After installation, activate the agkelpie environment
- Download an agkelpie dataset and set the appropriate folders and file locations. Note that by default, the code looks for `image_dir` in `annotated_images` from `annotated_images.txt` from the dataset.zip in order to differentiate (stay clear) of unannotated images.
    - `annotation_data`: a python dictionary with `annotation_file`, `image_dir` and `mask_dir` keys and corresponding values as strings for the absolute filepaths, respectively. The `annotation_file` is the string to the dataset.json automatically downloaded and packaged with the dataset. `image_dir` is the annotated image directory, and `mask_dir` is the directory of masks that have a binary representation of the annotations.
    - `output_dir`: the directory where models and progress checkpoints are saved during training
    - `classes_config`: a configuration file for class names and colours 
    - `imagelist_files`: a python dictionary with `train_file`, `val_file`, `test_file` keys that denote the corresponding training, validation and testing images that are saved in a list as textfiles (each image as a new line)
    - `hyper_parameters`: a json file containing the dictionary for a variety of hyper parameters for the training pipeline. See `TrainMaskRCNN.py` top-level comments for more information
- In `model_training.py`, set the appropriate folders and file locations: `annotation_data`, `output_dir`, `imagelist_files` and hyper-parameters. NOTE that we can easily set the annotation_data and imagelist_files by setting `root_dir` to the root directory of the dataset (which by default has been set to `agkelpiedataset_clarkefield31`), and `code_dir` to the repository root directory (i.e. the `weed-detection` folder).
- Run the top-level script, `model_training.py`, which calls `TrainMaskRCNN.train_pipeline()`
- In the specified `output_dir` (default is `/model/<dataset_name>/`), there will be several `.pth` files, which contain the model weights. 
    - The `model_best.pth` file is the model which corresponds to the lowest validation error during training. 
    - The `model_maxepochs.pth` file is the model which corresponds to the weights at the end of the maximum number of epochs, if early stopping did not trigger. If early stopping did trigger, `model_best.pth` is the final output model


## Running the Detector and Comparing to Annotations
- After installation (see above), and downloading relevant images, activate `agkelpie` environment
- In `model_test.py`, set the annotation data, output directory, model_file and species_file.
    - `annotation_data`, `output_dir`, `model_file` are as before
    - `species_file` is a textfile that has lines corresponding to species that the detector should expect to see in the data, and the order of which should correspond to the order with which the labels were specified during `model_training` (e.g. for clarkefield31, labels are "{1:'Tussock', 2:'Saffron thistle'}", and therefore `names_clarkefield31.txt` should be first line: Tussock, second line Saffron thistle.).
- Run `model_test.py`, which will run the detector on the specified number of images (currently set `max_img` to user-settable number), and save the figures in the output_dir. Open up said figures to verify functionality.



## Model Evaluation
- Model evaluation code from WeedModel.py (pre-2023) is now depracated
- Instead, we use UNE's Model Evaluation Package. See https://github.com/AgKelpie/une_weed_package for more details on how to run. 



## Future Work
- learn how to do hyperparameter sweeps using WandB to tune hyperparameters and ultimately optimise some of the model's performance
- negative image and ratio selection
- handle multiple datasets and merge them
- evaluate multi-object detection models

