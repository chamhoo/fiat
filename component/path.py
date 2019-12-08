"""
auther: leechh
"""
import cv2
import os
import shutil
from glob import glob


def count(path, suffix='jpg'):
    return len(glob(os.path.join(path, f'*.{suffix}')))


def mkdir(path):
    if path is not None:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)


def mkshape(path, suffix='jpg'):
    return cv2.imread(glob(os.path.join(path,  f'*.{suffix}'))[0]).shape


def pseudo_random(a, b, seed):
    """
    Linear congruential generator
    - https://en.wikipedia.org/wiki/Linear_congruential_generator
    """
    m = 2 ** 32
    while True:
        nextseed = (a * seed + b) % m
        yield nextseed
        seed = nextseed