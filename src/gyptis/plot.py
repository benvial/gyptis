#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

# __all__ = ["colors", "pause", "plot", "plot_subdomains"]

import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dolfin.common.plotting import mesh2triang
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Affine2D
from matplotlib.tri import Triangulation

from . import dolfin
from .complex import *
from .utils.helpers import project_iterative

colors = dict(
    red=(210 / 255, 95 / 255, 95 / 255), green=(69 / 255, 149 / 255, 125 / 255)
)

plt.register_cmap(
    cmap=LinearSegmentedColormap.from_list(
        "gyptis", [colors["green"], (1, 1, 1), colors["red"]], N=100
    )
)
plt.register_cmap(
    cmap=LinearSegmentedColormap.from_list(
        "gyptis_r", [colors["red"], (1, 1, 1), colors["green"]], N=100
    )
)
plt.register_cmap(
    cmap=LinearSegmentedColormap.from_list(
        "gyptis_white", [(1, 1, 1), (1, 1, 1), (1, 1, 1)], N=100
    )
)

plt.register_cmap(
    cmap=LinearSegmentedColormap.from_list(
        "gyptis_black", [(0, 0, 0), (0, 0, 0), (0, 0, 0)], N=100
    )
)


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


def plot_real(fplot, ax=None, geometry=None, proj_space=None, colorbar=True, **kwargs):
    proj = proj_space is not None
    if "cmap" not in kwargs:
        kwargs["cmap"] = "RdBu_r"
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if proj:
        fplot = project_iterative(fplot, proj_space)
    plt.sca(ax)
    p = dolfin.plot(fplot, **kwargs)
    cbar = plt.colorbar(p) if colorbar else None
    kwargs.pop("cmap")
    if geometry:
        geometry.plot_subdomains(**kwargs)
    return p, cbar


def plot_complex(
    fplot, ax=None, geometry=None, proj_space=None, colorbar=True, **kwargs
):
    proj = proj_space is not None
    if ax is None:
        fig, ax = plt.subplots(1, 2)
    P, C = [], []
    for a, f in zip(ax, [fplot.real, fplot.imag]):
        p, cbar = plot_real(
            f,
            ax=a,
            geometry=geometry,
            proj_space=proj_space,
            colorbar=colorbar,
            **kwargs,
        )
        P.append(p)
        C.append(cbar)
    return P, C


def plot(fplot, **kwargs):
    if iscomplex(fplot):
        return plot_complex(fplot, **kwargs)
    else:
        return plot_real(fplot, **kwargs)


def pause(interval):
    backend = plt.rcParams["backend"]
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def check_plot_type(plottype, f):
    if plottype == "real":
        fplot = f.real
    elif plottype == "imag":
        fplot = f.imag
    elif plottype == "module":
        fplot = f.module
    elif plottype == "phase":
        fplot = f.phase
    else:
        raise (
            ValueError(
                f"wrong plot type {plottype}, choose between real, imag, module or phase"
            )
        )
    return fplot
