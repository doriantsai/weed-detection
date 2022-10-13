# script to automatically make the conda environment defined in agkelpie.yml
# cd to weed_detection
mamba env create -f config/agkelpie.yml

# since opencv-python doesn't seem to play nicely with conda install options
# since running in conda, sometimes get Qt error associated with opencv,
# so pip install opencv-python will work, but has errors popping up
# current attempt does this, but build may take a while
# UPDATE: removed, as now redundant with "dependencies: opencv" in the agkelpie.yml file
# pip install --no-binary opencv-python opencv-python

conda activate agkelpie

# for weed-camera interface, unsure if zeromq == zmq, stackoverflow seems
# to imply they are different
pip install zmq

# section where we install weed_detection locally using pip
# cd weed_detection
pip install -e .

