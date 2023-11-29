#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


import pytest

from gyptis.utils.parallel import *


def f(x, y, a=1, b="b"):
    print(f"f called with {x} {y} {a} {b}")
    y = x**2
    print(f"output y = {x}^2 = {y}")
    return y


@parloop(n_jobs=4)
def fpar(x, *args, **kwargs):
    return f(x, *args, **kwargs)


def test_para():
    x = [1, 2, 3, 4]
    y = [f(_, 9) for _ in x]
    ypar = fpar(x, 9)
    assert ypar == y


def test_mpi_print():
    mpi_print("Hello world!")
