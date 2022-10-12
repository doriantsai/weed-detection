#! /usr/bin/env/python3

"""
- printing/generating figures
- 
"""

class Plotter:

    def __init__(self, display=True, save=False):
        self.save = save
        self.display = display

        