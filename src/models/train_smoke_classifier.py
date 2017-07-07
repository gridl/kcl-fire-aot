#!/usr/bin/env python

'''
Train smoke classification for MODIS scenes
'''

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import matplotlib.pyplot as plt
import logging

import src.config.filepaths as filepaths
import src.config.models as model_settings

__author__ = "Daniel Fisher"
__email__ = "daniel.fisher@kcl.ac.uk"


def main():

    # load the dataframe
    df = pd.read_pickle(filepaths.path_to_plume_classification_features)

    # do the PCA
    hold =1

    # do the training



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()