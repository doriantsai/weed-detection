#! /usr/bin/env python

"""
Attempt to get featurees out of deep weeds classifier model using register hooks
"""

# file locations (images/labels/model)
# import model
# feature extractor
# run model forward pass

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
# import torch.nn as nn

from torch import nn, Tensor
from torchvision.models import resnet, resnet50
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from typing import Dict, Iterable, Callable

# from https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904

# class FeatureExtractor()

# wrapper that prints the output shapes of each layer's output
class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

## example code to run/print layer shapes of resnet50
verbose_resnet = VerboseExecution(resnet50())
dummy_input = torch.ones(10, 3, 224, 224)
_ = verbose_resnet(dummy_input)

# wrapper that extracts features

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features


## example code to run/print feature of resenet50 layer4 and avgpool
resnet_features = FeatureExtractor(resnet50(), layers=["layer4", "avgpool"])
features = resnet_features(dummy_input)
print({name: output.shape for name, output in features.items()})

import code
code.interact(local=dict(globals(), **locals()))