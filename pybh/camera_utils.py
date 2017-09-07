import numpy as np
from contrib import transformations


def focal_length_to_fov(focal_length, length):
    """Convert focal length to field-of-view (given length of screen)"""
    fov = 2 * np.arctan(length / (2 * focal_length))
    return fov


def fov_to_focal_length(fov):
    """Convert field-of-view to focal length (given length of screen)"""
    focal_length = length / (2 * np.tan(fov / 2.))
    return focal_length
