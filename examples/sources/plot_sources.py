#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.3
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
Sources
========

Showcase of the various sources implemented in gyptis.
"""


# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy

##############################################################################
# Create the geometry and mesh

wl = 400
lmin = wl / 8
lbox = 5 * wl
pml_width = wl
geom = gy.BoxPML(
    dim=2,
    box_size=(lbox, lbox),
    pml_width=(pml_width, pml_width),
)
geom.set_size("box", lmin)
geom.set_pml_mesh_size(lmin)
output = geom.build()

plt.close("all")
plt.ion()
figsize = (4, 1.6)


def plot_arrow(a, x, y, dx, dy):
    a.arrow(
        x,
        y,
        dx,
        dy,
        width=wl / 100,
        head_width=wl / 5,
        color="y",
        length_includes_head=True,
    )


##############################################################################
# Plane wave
# ------------

pw = gy.PlaneWave(
    wavelength=wl,
    angle=np.pi / 3,
    dim=2,
    phase=3 * np.pi / 7,
    amplitude=0.1,
    domain=geom.mesh,
    degree=2,
)
fig, ax, plots, cbars = pw.plot(figsize)
for a in ax:
    x, y = 0, 0
    dx = -np.sin(pw.angle) * wl
    dy = -np.cos(pw.angle) * wl
    x0 = dx * 0.5
    y0 = dy * 0.5
    plot_arrow(a, x - x0, y - y0, dx, dy)


##############################################################################
# Line source
# ------------

ls = gy.LineSource(
    wavelength=wl,
    position=(wl, -wl * 1.4),
    dim=2,
    phase=np.pi / 3,
    amplitude=30,
    domain=geom.mesh,
    degree=2,
)
fig, ax, plots, cbars = ls.plot(figsize)
for a in ax:
    a.plot(*ls.position, "oy")


##############################################################################
# Dipole
# ------

dp = gy.Dipole(
    wavelength=wl,
    position=(wl / 2, -wl),
    angle=5 * np.pi / 6,
    phase=np.pi / 9,
    dim=2,
    domain=geom.mesh,
    degree=2,
)
fig, ax, plots, cbars = dp.plot(figsize)
for a in ax:
    a.plot(*dp.position, "oy")
    x, y = dp.position
    dx = -np.sin(dp.angle) * wl
    dy = -np.cos(dp.angle) * wl
    x0 = dx * 0.5
    y0 = dy * 0.5
    plot_arrow(a, x - x0, y - y0, dx, dy)


##############################################################################
# Gaussian beam
# ---------------

gb = gy.GaussianBeam(
    wavelength=wl,
    angle=np.pi / 9,
    position=(0.8 * wl, -wl * 0.3),
    dim=2,
    waist=0.5 * wl,
    phase=np.pi / 7,
    domain=geom.mesh,
    degree=2,
)
fig, ax, plots, cbars = gb.plot(figsize)
for a in ax:
    a.plot(*gb.position, "oy")
    x, y = gb.position
    dx = -np.sin(gb.angle) * wl
    dy = -np.cos(gb.angle) * wl
    x0 = dx * 2
    y0 = dy * 2
    plot_arrow(a, x - x0, y - y0, dx, dy)
