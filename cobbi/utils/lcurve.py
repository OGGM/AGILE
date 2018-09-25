import torch
import numpy as np


def create_logspaced_lamb_sequence(lamb_max, lamb_min, n):
    # Follows http://dx.doi.org/10.5772/intechopen.73332
    # Voronin, Zaroli 2018
    # "Survey of Computational Methods for Inverse Problems"
    S = (np.log(lamb_max) - np.log(lamb_min)) / (n - 1)
    lambs = np.exp(np.log(lamb_max) - S * np.arange(0, n))
    return lambs