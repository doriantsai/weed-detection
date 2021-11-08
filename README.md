# weed-detection
A work-in-progress machine learning approach to weed detection for the agkelpie project

File and Folder Descriptions:

- weed_detection - fine-tuned deteector for tussock with model training, model evaluation scripts
- scripts - a set of scripts used to run certain operations
- x_depracated - a set of old scripts that may no longer work and should be removed from the repo
- agkelpie.yml - the conda environment script file
- make_conda_agkelpie.sh - the script that automatically creates the agkelpie conda virtual environment


QUT agkelpie weed detector deployment model

Currently, conda is used to create a virtual python environment for the agkelpie
weed detector to operate within.

Installation:
- Install conda https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
- git clone git@github.com:doriantsai/agkelpie-weed-detector.git
- cd to agkelpie-weed-detector folder
- Create conda environment automatically by running the script "make_conda_agkelpie.sh" file
    - make .sh file executable via "chmod +x make_conda_agkelpie.sh"
    - run .sh file via "./make_conda_agkelpie.sh"
- this should automatically make a virtual environment called "agkelpie"
- activate new environment: "conda activate agkelpie"

Alternatively:
- alternatively, create a new conda environment: conda create -n "agkelpie" python=3.8
- activate new environment: conda activate agkelpie
- install packages listed in agkelpie.yml dependencies using conda install X, or pip install X
- as per the commands in make_conda_agkelpie.sh file:
- pip install --no-binary opencv-python opencv-python
- pip install zmq (for weed-camera interface)
- pip install -e . (installs the agkelpie-weed-detector as a package)

- ensure agkelpie virtual environment is working (should see "agkelpie")
- Run "python working_example.py", see output in output folder

