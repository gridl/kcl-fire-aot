#!/usr/bin/env python

'''
Using the smoke random forest smoke classifier process all the MODIS data that we
have and generate smoke plume masks.

- Load in MODIS data
- Derive GLCM for MODIS data
- Apply smoke classifier
- Save mask output in
'''

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import logging

import matplotlib.pyplot as plt

import resampling
import src.config.filepaths as filepaths
import src.config.features as features_settings
import src.data.readers as readers
import GLCM.Textures as textures

__author__ = "Daniel Fisher"
__email__ = "daniel.fisher@kcl.ac.uk"


def main():
    pass


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()