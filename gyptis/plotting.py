#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import matplotlib.pyplot as plt
import numpy as np
from dolfin.common.plotting import mesh2triang
from matplotlib.tri import Triangulation

plt.ion()


def get_bnds(markers):

    data = markers.array()
    triang = mesh2triang(markers.mesh())
    ids = np.unique(data)

    import copy

    triangulations = []
    for id in ids:
        triang_ = copy.deepcopy(triang)
        triang_.set_mask(data != id)
        triangulations.append(triang_)

    sub_bnds = []
    for triangtest in triangulations:

        # triangtest=triangulations[2]

        maskedTris = triangtest.get_masked_triangles()
        verts = np.stack((triangtest.x[maskedTris], triangtest.y[maskedTris]), axis=-1)
        all_vert = np.vstack(verts).T
        # unique_vert = np.array(list(set(tuple(p) for p in all_vert.T))).T
        # masked_triangles = triangtest.triangles[~triangtest.mask]
        sub_triang = Triangulation(*all_vert)

        boundaries = []
        for i in range(len(sub_triang.triangles)):
            for j in range(3):
                if sub_triang.neighbors[i, j] < 0:
                    # Triangle edge (i,j) has no neighbor so is a boundary edge.
                    boundaries.append(
                        (
                            sub_triang.triangles[i, j],
                            sub_triang.triangles[i, (j + 1) % 3],
                        )
                    )
        boundaries = np.asarray(boundaries)

        bndpnts = sub_triang.x[boundaries].T, sub_triang.y[boundaries].T

        sub_bnds.append(bndpnts)

    return sub_bnds


def plot_lines(markers, ax=None, **kwargs):
    sub_bnds = get_bnds(markers)
    if ax is None:
        ax = plt.gca()
    l = []
    for bndpnts in sub_bnds:
        l_ = ax.plot(*bndpnts, **kwargs)
        l.append(l_)
    return l


def plot_subdomains(markers, alpha=0.3):
    a = df.plot(markers, cmap="binary", alpha=alpha, lw=0.00, edgecolor="face")
    return a
    # a.set_edgecolors((0.1, 0.2, 0.5, 0.))


def plot_subdomains(markers, alpha=1, **kwargs):
    return plot_lines(markers, ax=None, color="k", **kwargs)
    # a.set_edgecolors((0.1, 0.2, 0.5, 0.))


def plotcplx(test, ax, markers=None, W0=None, ref_cbar=False, **kwargs):

    proj = W0 is not None

    if proj:
        test = project(test, W0)
    plt.sca(ax[0])
    p = df.plot(test.real, cmap="RdBu_r", **kwargs)
    cbar = plt.colorbar(p)
    if markers:
        plot_subdomains(markers)
    if ref_cbar:
        v = test.real.vector().get_local()
        mn, mx = min(v), max(v)
        md = 0.5 * (mx + mn)
        cbar.set_ticks([mn, md, mx])
        lab = [f"{m:.2e}" for m in [mn, md, mx]]
        cbar.set_ticklabels(lab)
    plt.sca(ax[1])
    p = df.plot(test.imag, cmap="RdBu_r", **kwargs)
    cbar = plt.colorbar(p)
    if markers:
        plot_subdomains(markers)
    if ref_cbar:
        v = test.imag.vector().get_local()
        mn, mx = min(v), max(v)
        md = 0.5 * (mx + mn)
        cbar.set_ticks([mn, md, mx])
        lab = [f"{m:.2e}" for m in [mn, md, mx]]
        cbar.set_ticklabels(lab)
    return p, cbar


def plot(test, W0=None, markers=None, **kwargs):
    proj = W0 is not None
    if proj:
        test = project(test, W0)
    p = df.plot(test, cmap="inferno", **kwargs)
    cbar = plt.colorbar(p)
    if markers:
        plot_subdomains(markers)

    return p, cbar
