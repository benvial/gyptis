#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


"""
This module provides various source implementations for electromagnetic simulations in the gyptis library.

The sources include different types of electromagnetic waveforms such as dipole, Gaussian beam, line source,
plane wave, and spherical harmonics. These sources are essential for defining the incident fields in
computational simulations using the gyptis framework.

Modules:
- dipole: Defines a Dipole source for 2D electromagnetic simulations.
- gaussian: Implements a Gaussian beam source.
- ls: Contains the LineSource class for representing line sources.
- pw: Provides the PlaneWave class for plane wave sources.
- source: Abstract base class for all sources, defining common properties and methods.
- spharm: Implements spherical harmonics and related operations.
- stack: Utilities for stacking and manipulating sources.

Each module is designed to be used in the context of finite element method (FEM) simulations, with support
for 2D and partially for 3D geometries.
"""

from .dipole import *
from .gaussian import *
from .ls import *
from .pw import *
from .source import *
from .spharm import *
from .stack import *
