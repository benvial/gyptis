# -*- coding: utf-8 -*-
"""
Band diagram of 2D photonic crystal
===================================

Calculation of the band diagram of a two-dimensional photonic crystal.
"""


# sphinx_gallery_thumbnail_number = -1

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

from gyptis import Lattice, PhotonicCrystal
from gyptis.complex import project

##############################################################################
# Reference results are taken from [Joannopoulos2008]_ (Chapter 5 Fig. 2).
#
# The structure is a square lattice of dielectric
# columns, with radius r and dielectric constant :math:`\varepsilon`.
# The material is invariant along the z direction  and periodic along
# :math:`x` and :math:`y` with lattice constant :math:`a`.
# We will define the geometry using the class :class:`~gyptis.Lattice`:
# We will define the geometry using the class :class:`~gyptis.Geometry`:


######################################################################
#
# .. [Joannopoulos2008] Joannopoulos, J. D., Johnson, S. G., Winn, J. N., & Meade, R. D.,
#    Photonic Crystals: Molding the flow of light.
#    Princeton Univ. Press, Princeton, NJ, (2008).
#    `<http://ab-initio.mit.edu/book/>`_
