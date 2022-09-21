# weed-detection
A deep learning approach to pastoral weed detection. 

# General info
A deep learning approach to weed species detection for smart farm pasture management developed by QUT, in collaboration with AOS, DPI and UNE. With a Resnet50 backbone, FasterRCNN and MaskRCNN are used for bounding box and polygon annotations, respectively. A virtual environment is setup to contain the weed detection model via conda.

# Data
Data is obtained from the DPI's Weed Reference Library, currently stored on AWS. Access is currently limited for commercial development reasons. See the following link for the AWS workshop that Elrond Cheung conducted to help setup access. The current version of the weed detector only considers image data that has been organised into the proper date and location schema in the folder **03_Tagged**.
https://help.agkelpie.com/aws_workshop.html, specifically, see the AWS workshop pdf for detailed instructions on the setup.
https://help.agkelpie.com/AgKelpieImageDatabaseAWSWorkshop.pdf

To download data via script, you need to setup AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    aws configure

Use `us-east-1` as the region, and `json` as the output. I also had to use my full mfa device id `arn:aws:iam::############:mfa/FirstName.LastName`, and not just `::############:mfa/FirstName.LastName` as the tutorial indicated.

    aws sts get-session-token --serial-number XX --token-code YY
    
# File and Folder Descriptions
* **weed_detection** - fine-tuned deteector for tussock with model training, model evaluation scripts
* **examples** - a minimum working example for model inference with a sample image, designed to work with the Agkelpie weed_camera_interface code available at https://github.com/AgKelpie/weed_camera_interface
* **scripts** - a set of scripts used to run various operations, eg. processing the dataset, training the model, generating performance-recall curves. Very much in development and could use refactoring.
- agkelpie.yml - the conda environment script file
- make_conda_agkelpie.sh - the script that automatically creates the agkelpie conda virtual environment
- setup.py - along with __init__.py in weed_detection, required to install ``weed_detection`` as a python package


# Installation (script install)
- Install conda https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html, although I recommend miniforge/mambaforge https://github.com/conda-forge/miniforge, which is a much more minimal and faster installer for conda packages. These installers will create a ''base'' environment that contains the package managers for conda.
- ```conda install mamba -c conda-forge``` 
- ```git clone git@github.com:doriantsai/weed-detection.git```
- ```cd``` to agkelpie-weed-detector folder
- Create conda environment automatically by running the script "make_conda_agkelpie.sh" file
    - make .sh file executable via ```chmod +x make_conda_agkelpie.sh```
    - run .sh file via ```./make_conda_agkelpie.sh```
- this should automatically make a virtual environment called "agkelpie"
- to activate new environment: ```conda activate agkelpie```
- to deactivate the new environment: ```conda deactivate```

## Alternatively (manual install)
- alternatively, a more manual process is to create a new conda environment: ```conda create -n "agkelpie" python=3.8```
- activate new environment: ```conda activate agkelpie```
- install packages listed in agkelpie.yml dependencies using ```conda install X```, or ```pip install X```
- as per the commands in make_conda_agkelpie.sh
- ensure agkelpie virtual environment is working (```conda env list```)

# Example
- Install weed_detection module, as per the **Installation** instructions above
- Activate ```agkelpie``` environment
- navigate to examples ```cd examples```
- Run ```python working_example.py``` and see output in ```output``` folder

