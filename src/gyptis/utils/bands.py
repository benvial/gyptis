#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

from itertools import accumulate

import matplotlib.pyplot as plt
import numpy as np


def init_bands(sym_points, nband):
    path = []
    num_point = len(sym_points)
    for i in range(num_point - 1):
        P = sym_points[i]
        Q = sym_points[i + 1]
        _kx = np.linspace(P[0], Q[0], nband)
        _ky = np.linspace(P[1], Q[1], nband)
        k = np.vstack([_kx, _ky]).T
        if i != num_point - 2:
            k = k[:-1]
        path.append(k)
    ks = np.vstack(path)
    return ks


def init_bands_plot(sym_points, nband):
    ks = init_bands(sym_points, nband)
    dk = []
    for ik in range(len(ks) - 1):
        dk.append(np.linalg.norm(ks[ik + 1] - ks[ik]))
    ksplot = np.array(list(accumulate([0] + dk)))
    dk = []
    for ik in range(len(sym_points) - 1):
        dk.append(
            np.linalg.norm(np.array(sym_points[ik + 1]) - np.array(sym_points[ik]))
        )
    ksym = np.array(list(accumulate([0] + dk)))
    return ksplot, ksym


def plot_bands(
    sym_points,
    nband,
    eigenvalues,
    xtickslabels=None,
    color=None,
    **kwargs,
):
    if color == None:
        color = "#4d63c5"
        if "color" in kwargs:
            kwargs.pop("colors")
        if "c" in kwargs:
            kwargs.pop("c")
    ksplot, ksym = init_bands_plot(sym_points, nband)
    plt.plot(ksplot, eigenvalues, color=color, **kwargs)
    if xtickslabels is not None:
        plt.xticks(ksym, xtickslabels)
        for k in ksym:
            plt.axvline(k, c="#8a8a8a")
    plt.xlim(ksym[0], ksym[-1])
    plt.ylim(0)
    plt.ylabel(r"$\omega$")
