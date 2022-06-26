#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to create plots for use in LaTeX documents.

Author:  Christopher Good
Version: 1.0.0

Usage: plot_creator.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###
import argparse
import os

### Other Library Imports ###
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

### Local Imports ###
from data.grss_dfc_2018_uh import UH_2018_Dataset

### Environment ###

# Set up matplotlib to save figures in pgf format for
# using in LaTeX
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

### Constants ###
#TODO

### Definitions ###

def create_and_save_bar_plot(file_save_path, data, 
                             plot_title, x_axis_label, y_axis_label, 
                             figure_size=(12,9), **kwargs):
    
    fig = plt.figure(figsize=figure_size)
    plt.bar()

### Main ###
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('LaTeX plot creation script')