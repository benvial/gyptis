#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import pytest
from gyptis.sampling import adaptive_sampler
import numpy as np

def test_sampler():
    # np.random.seed(123456)

    Npoles = 100
    poles = np.random.rand(Npoles) + 1j * np.random.rand(Npoles)
    res = np.random.rand(Npoles) + 1j * np.random.rand(Npoles)

    def f(z):
        t = 0
        for p, r in zip(poles, res):
            t += r / (z - p)
        Q = np.abs(t) ** 2

        return Q / Npoles ** 2

    npts = 2000
    z_ = np.linspace(0, 1, npts)
    Q = f(z_)

    # adapt

    zmin, zmax = 0, 1
    n0 = 33
    z0 = np.linspace(zmin, zmax, n0)
    # t0 = f(z0)

    z, t = adaptive_sampler(f, z0)
    t0 = f(z0)

    print(f"number of points: {len(z)}")
