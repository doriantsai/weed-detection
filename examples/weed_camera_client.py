# Test code to receive open cv images

import zmq
import cv2 as cv
import numpy
import os
from weed_detection.WeedModel import WeedModel
import time
from subprocess import call

#########################################################
# Client to connect to the weed camera server using ZMQ
# Currently only raw CVMat reading is supported in the Python version
# The image height and width values will be sent as part of the protocol in the future
class WeedCameraClient:
    def __init__(self, address_string = "tcp://127.0.0.1:5556", img_height = 2056, img_width = 2464):
        self.address_string = address_string
        self.img_height = img_height
        self.img_width = img_width

    # Subscribe to images
    def connect(self):
        print("Connecting to server...")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')    # Subscribe to all messages
        self.socket.setsockopt(zmq.CONFLATE, 1)             # Last message only
        self.socket.connect(self.address_string)

    # Read a single image - must connect first
    def read_image(self):
        # Grab image
        input_buffer = self.socket.recv()
        # Convert buffer to numpy array
        image = numpy.frombuffer(input_buffer, dtype=numpy.uint8).reshape((self.img_height, self.img_width, 3))
        # Return value
        return image

#########################################################


# setup weed model
# model_name = '2021-03-25_MFS_Tussock_v0_2021-09-16_08_55'
# model_path = os.path.join('/home/dorian/Code/weed-detection/scripts/output', model_name, model_name + '.pth')
# Tussock.load_model(model_path)
# Tussock.set_model_name(model_name)
# Tussock.set_snapshot(25)
# check if model is in folder. If so, use it. Otherwise, download it
model_path = os.path.join('models','detection_model.pth')
if not os.path.exists(model_path):
    # download model to folder/file-path specified by model_path
    os.makedirs('models', exist_ok=True)
    url = 'https://cloudstor.aarnet.edu.au/plus/s/ZJQAKiOZFDBxDJc/download'
    call(['wget', '-O', model_path, url])

# setup weed model
Tussock = WeedModel()
Tussock.load_model(model_path)

# Demonstration code to run as script
# Create client using default arguments
client = WeedCameraClient()
# Connect and subscribe to image
client.connect()
# Loop and continually grab the latest image and display it

start_time = time.time()
i = 0
MAX_COUNT = 50
while(i < MAX_COUNT):
    print(f'{i}: read image')
    image = client.read_image()

    # TODO red blue green channels are mixed up!
    print('   model inference')
    img_name = 'image_' + str(i)
    img_out, pred = Tussock.infer_image(image,
                                        imshow=False,
                                        imsave=True,
                                        save_dir='.',
                                        image_name=img_name)

    i += 1

# print times
end_time = time.time()
sec = end_time - start_time
cycle_time = sec / i
print('running time: {} sec'.format(sec))
print('cycle time: {} sec/cycle'.format(cycle_time))
print('cycle freq: {} Hz'.format(1.0/cycle_time))
