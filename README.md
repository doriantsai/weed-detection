# weed-detection
A work-in-progress deep learning approach to weed detection for the agkelpie project

# General info
A deep learning approach to weed species detection for smart farm pasture management developed by QUT, in collaboration with AOS, DPI and UNE. With a Resnet50 backbone, FasterRCNN and MaskRCNN are used for bounding box and polygon annotations, respectively. A virtual environment is setup to contain the weed detection model via conda.

# Data
TODO: describe folder setup - hyperlinks to image folder data folders and hierarchy.

# File and Folder Descriptions
* **weed_detection** - fine-tuned deteector for tussock with model training, model evaluation scripts
* **examples** - a minimum working example for model inference with a sample image
* **scripts** - a set of scripts used to run various operations, eg. processing the dataset, training the model, generating performance-recall curves
- x_depracated - a set of old scripts that may no longer work and should be removed from the repo
- agkelpie.yml - the conda environment script file
- make_conda_agkelpie.sh - the script that automatically creates the agkelpie conda virtual environment

# Installation
- Install conda https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
- ```git clone git@github.com:doriantsai/agkelpie-weed-detector.git```
- ```cd``` to agkelpie-weed-detector folder
- Create conda environment automatically by running the script "make_conda_agkelpie.sh" file
    - make .sh file executable via ```chmod +x make_conda_agkelpie.sh```
    - run .sh file via ```./make_conda_agkelpie.sh```
- this should automatically make a virtual environment called "agkelpie"
- activate new environment: ```conda activate agkelpie```

## Alternatively
- alternatively, a more manual process is to create a new conda environment: ```conda create -n "agkelpie" python=3.8```
- activate new environment: ```conda activate agkelpie```
- install packages listed in agkelpie.yml dependencies using ```conda install X```, or ```pip install X```
- as per the commands in make_conda_agkelpie.sh file:
```
$ pip install --no-binary opencv-python opencv-python
$ pip install zmq
$ pip install -e .
```
- ensure agkelpie virtual environment is working (should see "agkelpie" in the terminal

# Example
- Install weed_detection module, as per the **Installation** instructions
- Activate ```agkelpie``` environment
- navigate to examples ```cd examples```
- Run ```python working_example.py``` and (hopefully) see output in output folder

