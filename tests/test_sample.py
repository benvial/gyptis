#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

import numpy as np
import pytest

from gyptis.utils.sample import _adaptive_sampler, adaptive_sampler

np.random.seed(13)


def test_sampler():
    Npoles = 100

    poles = np.random.rand(Npoles) + 1j * np.random.rand(Npoles)
    res = np.random.rand(Npoles) + 1j * np.random.rand(Npoles)

    def f(z):
        t = 0
        for p, r in zip(poles, res):
            t += r / (z - p)
        Q = np.abs(t) ** 2
        return Q / Npoles**2

    npts = 2000
    zref = np.linspace(0, 1, npts)
    f(zref)

    # adapt

    zmin, zmax = 0, 1
    n0 = 33
    z0 = np.linspace(zmin, zmax, n0)
    # t0 = f(z0)

    z, t = _adaptive_sampler(f, z0)
    f(z0)

    print(f"number of points: {len(z)}")
    print("------------")

    def f1(z):
        return f(z), z**2

    z, t = _adaptive_sampler(f1, z0)
    f(z0)
    assert len(z) == len(t)

    print(f"number of points: {len(z)}")

    @adaptive_sampler()
    def f2(z, a, b="test"):
        print(a)
        print(b)
        return f1(z)

    z1, t1 = f2(z0, 10, b="hello")
    assert len(z1) == len(t1)
    assert np.allclose(z1, z)
    assert np.allclose(t1, t)
