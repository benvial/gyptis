#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import copy

import matplotlib.pyplot as plt
import numpy as np
from dolfin.common.plotting import mesh2triang
from matplotlib.tri import Triangulation

from gyptis.complex import *

from . import dolfin


def get_boundaries(markers, domain=None, shift=(0, 0)):

    data = markers.array()
    triang = mesh2triang(markers.mesh())
    if domain == None:
        ids = np.unique(data)
    else:
        ids = [domain]

    triangulations = []
    for id in ids:
        triang_ = copy.deepcopy(triang)
        triang_.set_mask(data != id)
        triangulations.append(triang_)

    sub_bnds = []
    for triangtest in triangulations:
        maskedTris = triangtest.get_masked_triangles()
        verts = np.stack((triangtest.x[maskedTris], triangtest.y[maskedTris]), axis=-1)
        all_vert = np.vstack(verts).T
        sub_triang = Triangulation(*all_vert)

        boundaries = []
        for i in range(len(sub_triang.triangles)):
            for j in range(3):
                if sub_triang.neighbors[i, j] < 0:
                    boundaries.append(
                        (
                            sub_triang.triangles[i, j],
                            sub_triang.triangles[i, (j + 1) % 3],
                        )
                    )
        boundaries = np.asarray(boundaries)

        bndpnts = (
            shift[0] + sub_triang.x[boundaries].T,
            shift[1] + sub_triang.y[boundaries].T,
        )

        sub_bnds.append(bndpnts)

    return sub_bnds


def plot_boundaries(markers, domain=None, shift=(0, 0), ax=None, **kwargs):
    sub_bnds = get_boundaries(markers, domain=domain, shift=shift)
    if "c" not in kwargs and "color" not in kwargs:
        kwargs["color"] = "k"
    if ax is None:
        ax = plt.gca()
    l = []
    for bndpnts in sub_bnds:
        l_ = ax.plot(*bndpnts, **kwargs)
        l.append(l_)
    return l


# def plot_subdomains(markers, alpha=0.3):
#     a = dolfin.plot(markers, cmap="binary", alpha=alpha, lw=0.00, edgecolor="face")
#     return a
#     # a.set_edgecolors((0.1, 0.2, 0.5, 0.))


def plot_subdomains(markers, **kwargs):
    return plot_boundaries(markers, **kwargs)


def plotcplx(test, ax=None, markers=None, proj_space=None, ref_cbar=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 2)

    if "cmap" not in kwargs:
        kwargs["cmap"] = "RdBu_r"
    proj = proj_space is not None

    if proj:
        test = project(test, proj_space)
    P, C = [], []
    for a, t in zip(ax, [test.real, test.imag]):
        plt.sca(a)
        p = dolfin.plot(t, **kwargs)
        cbar = plt.colorbar(p)
        if markers:
            plot_subdomains(markers, **kwargs)
        if ref_cbar:
            v = test.real.vector().get_local()
            mn, mx = min(v), max(v)
            md = 0.5 * (mx + mn)
            cbar.set_ticks([mn, md, mx])
            lab = [f"{m:.2e}" for m in [mn, md, mx]]
            cbar.set_ticklabels(lab)
        P.append(p)
        C.append(cbar)
    return P, C


def plot(test, ax=None, markers=None, proj_space=None, **kwargs):
    proj = proj_space is not None
    if "cmap" not in kwargs:
        kwargs["cmap"] = "inferno"
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if proj:
        test = project(test, proj_space)
    plt.sca(ax)
    p = dolfin.plot(test, **kwargs)
    cbar = plt.colorbar(p)
    if markers:
        plot_subdomains(markers)

    return p, cbar
