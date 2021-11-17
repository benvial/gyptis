#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import numpy as np
import pytest

from gyptis.utils.sample import adaptive_sampler


def test_sampler():
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
    zref = np.linspace(0, 1, npts)
    tref = f(zref)

    # adapt

    zmin, zmax = 0, 1
    n0 = 33
    z0 = np.linspace(zmin, zmax, n0)
    # t0 = f(z0)

    z, t = adaptive_sampler(f, z0)
    t0 = f(z0)

    print(f"number of points: {len(z)}")
    # xsa
    print("------------")

    def f(z):
        t = 0
        for p, r in zip(poles, res):
            t += r / (z - p)
        Q = np.abs(t) ** 2
        return Q / Npoles ** 2, z ** 2

    z, t = adaptive_sampler(f, z0)
    t0 = f(z0)

    print(f"number of points: {len(z)}")
