import numpy as np

def rmse(a1, a2):
    return np.sqrt(np.mean((a1 - a2)**2))