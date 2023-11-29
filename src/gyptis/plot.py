#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


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

if "gyptis" not in plt.colormaps():
    matplotlib.colormaps.register(
        cmap=LinearSegmentedColormap.from_list(
            "gyptis", [colors["green"], (1, 1, 1), colors["red"]], N=100
        )
    )
if "gyptis_r" not in plt.colormaps():
    matplotlib.colormaps.register(
        cmap=LinearSegmentedColormap.from_list(
            "gyptis_r", [colors["red"], (1, 1, 1), colors["green"]], N=100
        )
    )

if "gyptis_white" not in plt.colormaps():
    matplotlib.colormaps.register(
        cmap=LinearSegmentedColormap.from_list(
            "gyptis_white", [(1, 1, 1), (1, 1, 1), (1, 1, 1)], N=100
        )
    )

if "gyptis_black" not in plt.colormaps():
    matplotlib.colormaps.register(
        cmap=LinearSegmentedColormap.from_list(
            "gyptis_black", [(0, 0, 0), (0, 0, 0), (0, 0, 0)], N=100
        )
    )


def get_boundaries(markers, domain=None, shift=(0, 0)):
    data = markers.array()
    triang = mesh2triang(markers.mesh())
    ids = np.unique(data) if domain is None else [domain]
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
            boundaries.extend(
                (
                    sub_triang.triangles[i, j],
                    sub_triang.triangles[i, (j + 1) % 3],
                )
                for j in range(3)
                if sub_triang.neighbors[i, j] < 0
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
    lines = []
    for bndpnts in sub_bnds:
        line = ax.plot(*bndpnts, **kwargs)
        lines.append(line)
    return lines


# def plot_subdomains(markers, alpha=0.3):
#     a = dolfin.plot(markers, cmap="binary", alpha=alpha, lw=0.00, edgecolor="face")
#     return a
#     # a.set_edgecolors((0.1, 0.2, 0.5, 0.))


def plot_markers(markers, subdomains, ax=None, geometry=None, colorbar=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if "cmap" not in kwargs:
        kwargs["cmap"] = "inferno"
    plt.sca(ax)
    n = len(subdomains)
    cmap_disc = plt.cm.get_cmap(kwargs["cmap"])
    bounds = np.array(list(subdomains.values()))
    srt = np.argsort(bounds)
    ids = np.array(list(subdomains.keys()))[srt]
    bounds = bounds[srt]
    bounds = np.hstack([bounds, [bounds[-1] + 1]])

    tt = [(bounds[i + 1] + bounds[i]) / 2 for i in range(n)]

    norm = matplotlib.colors.BoundaryNorm(bounds, cmap_disc.N)
    kwargs["cmap"] = cmap_disc
    p = dolfin.plot(markers, norm=norm, **kwargs)
    if colorbar:
        cbar = plt.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_disc), ticks=tt, ax=ax
        )

        cbar.ax.set_yticklabels(ids)
        cbar.ax.tick_params(length=0)
    else:
        cbar = None
    kwargs.pop("cmap")
    # cbar = plt.colorbar(p,ticks=list(subdomains.keys()),cmap=cmap_disc) if colorbar else None

    if geometry:
        geometry.plot_subdomains(**kwargs)
    return p, cbar


def plot_subdomains(markers, **kwargs):
    return plot_boundaries(markers, **kwargs)


def plot_real(
    fplot,
    ax=None,
    geometry=None,
    proj_space=None,
    colorbar=True,
    orientation="vertical",
    **kwargs,
):
    proj = proj_space is not None
    if "cmap" not in kwargs:
        kwargs["cmap"] = "RdBu_r"
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if proj:
        fplot = project_iterative(fplot, proj_space)
    plt.sca(ax)
    p = dolfin.plot(fplot, **kwargs)
    cbar = plt.colorbar(p, orientation=orientation) if colorbar else None
    kwargs.pop("cmap")
    if geometry:
        geometry.plot_subdomains(**kwargs)
    return p, cbar


def plot_complex(
    fplot,
    ax=None,
    geometry=None,
    proj_space=None,
    colorbar=True,
    orientation="vertical",
    **kwargs,
):
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
            orientation=orientation,
            **kwargs,
        )
        P.append(p)
        C.append(cbar)
    return P, C


def plot(fplot, ax=None, geometry=None, proj_space=None, colorbar=True, **kwargs):
    returnfunc = plot_complex if iscomplex(fplot) else plot_real
    return returnfunc(fplot, ax, geometry, proj_space, colorbar, **kwargs)


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
