#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import numpy as np

from gyptis.helpers import *


def test_all():
    mpi_print("Hello world!")
    matfmt(np.random.rand(3, 3) - 1j * np.random.rand(3, 3), cplx=True)
    assert tanh(1.2) == np.tanh(1.2)
