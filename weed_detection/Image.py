#! /usr/bin/env python3

"""
Image class, which has annotation properties, such as filename, width, camerea, height, regions
"""

from weed_detection.Region import Region

class Image:

    def __init__(self, 
                 filename: str,
                 file_attr: dict = None,
                 width: int = None,
                 height: int = None, 
                 camera: str = None):
        """__init__

        Args:
            filename (str): _description_
            file_attr (dict, optional): _description_. Defaults to None.
            width (int, optional): _description_. Defaults to None.
            height (int, optional): _description_. Defaults to None.
            camera (str, optional): _description_. Defaults to None.
        """        
        self.filename = filename 
        self.file_attr = file_attr
        self.width = width
        self.height = height
        self.camera = camera
        self.regions = []

    def print(self):
        """print
        """        
        print('Filename: ' + self.filename)
        # print(f'Filesize: {self.filesize}')
        print('File attributes:')
        print(self.file_attr)
        print(f'Image (width, height): ({self.width, self.height}) pix')
        print('Camera: ' + self.camera)
        print('Regions: ')
        for region in self.regions:
            region.print()
