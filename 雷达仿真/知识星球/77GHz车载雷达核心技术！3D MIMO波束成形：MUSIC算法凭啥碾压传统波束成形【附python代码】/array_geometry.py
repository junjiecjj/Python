# Defines the planar rectangular antenna array.

import numpy as np
from config import NX, NY, ELEMENT_SPACING

def generate_array_positions():

#    Generates 2D planar array coordinates

    x = np.arange(NX) * ELEMENT_SPACING
    y = np.arange(NY) * ELEMENT_SPACING

    xx, yy = np.meshgrid(x, y)

    positions = np.vstack((xx.ravel(), yy.ravel()))

    return positions