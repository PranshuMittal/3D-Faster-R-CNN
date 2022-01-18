"""
3D Faster R-CNN
Train on the CT scan of legs dataset
"""

import os
import sys
import json
import datetime
import numpy as np
import PIL

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

from faster_rcnn.config import Config
from faster_rcnn import model as modellib, utils

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')


############################################################
#  Configurations
############################################################

class CTLandmarkConfig(Config):

    """
    Configuration for training of the landmark dataset/
    Derive from the base config an overrides some methods
    """
    # Give the configuration a recognizable name
    NAME = 'ct_landmark'

    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 34  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CTLandmarkDataset(utils.Dataset):

    def load_scan(self, dataset_dir, subset):
       """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """ 