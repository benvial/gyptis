#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
Local density of states
=======================

Calculation of the Green's function and LDOS in 2D finite photonic crystals.
"""


# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy

##############################################################################
# Reference results are taken from :cite:p:`Asatryan2001`.

wavelength = 3.5
pmesh = 12  # mesh parameter
a = 0.3  # radius of the rods
n_cyl = 3  # index of the rods

##############################################################################
# We define the position of the centers of the rods:

rod_positions = [(i, j) for i in range(-3, 4) for j in range(-3, 4)]
rod_positions += [(i, j) for i in [-4, 4] for j in range(-3, 4)]
rod_positions += [(i, j) for j in [-4, 4] for i in range(-3, 4)]
rod_positions += [(i, 0) for i in [-5, 5]]
rod_positions += [(0, i) for i in [-5, 5]]


def plot_rods(ax, rod_positions):
    for pos in rod_positions:
        circle = plt.Circle(pos, a, fill=False)
        ax.add_patch(circle)


##############################################################################
# Build and mesh the geometry:


def create_geometry(wavelength, pml_width, group=False):
    lmin = wavelength / pmesh

    geom = gy.BoxPML(
        dim=2,
        box_size=(16, 16),
        pml_width=(pml_width, pml_width),
    )
    box = geom.box
    cylinders = []
    for pos in rod_positions:
        cyl = geom.add_circle(*pos, 0, a)
        cylinders.append(cyl)
    *cylinders, box = geom.fragment(cylinders, box)
    geom.add_physical(box, "box")
    [geom.set_size(pml, lmin * 1) for pml in geom.pmls]
    geom.set_size("box", lmin)
    if group:
        geom.add_physical(cylinders, "cylinders")
        geom.set_size("cylinders", lmin / n_cyl)
    else:
        # we could define physical domains for each rod but that is slower
        # when assembling and solving the sctattering problem
        for i, cyl in enumerate(cylinders):
            geom.add_physical(cyl, f"cylinder_{i}")
            geom.set_size(f"cylinder_{i}", lmin / n_cyl)
    geom.build()
    return geom


geom = create_geometry(wavelength, pml_width=wavelength, group=True)

##############################################################################
# Define the line excitation and materials

ls = gy.LineSource(wavelength=wavelength, position=(0, 7.3), domain=geom.mesh, degree=2)

epsilon = {d: n_cyl**2 for d in geom.domains}
epsilon["box"] = 1
mu = {d: 1 for d in geom.domains}

##############################################################################
# Instanciate and solve the scattering problem:

s = gy.Scattering(geom, epsilon, mu, ls, degree=2, polarization="TM")
s.solve()
G = s.solution["total"]

##############################################################################
# Here we project the function to plot on a suitable function space using an
# iterative solver:

v = gy.dolfin.ln(abs(G)) / gy.dolfin.ln(10)
vplot = gy.project(
    v,
    s.formulation.real_function_space,
    solver_type="cg",
    preconditioner_type="jacobi",
)

##############################################################################
# Plot the Green's function:

fig, ax = plt.subplots(figsize=(2.6, 2.2))
cs = gy.dolfin.plot(vplot, mode="contourf", cmap="Spectral_r", levels=31)
plot_rods(ax, rod_positions)
plt.axis("square")
plt.xlabel(r"$x/d$")
plt.ylabel(r"$y/d$")
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.xticks(np.arange(-8, 9, 4))
plt.yticks(np.arange(-8, 9, 4))
plt.colorbar(cs, fraction=0.04, pad=0.08)
plt.title(r"$\log_{10}|G\,|$")
plt.tight_layout()


##############################################################################
# Due to symmetry we will only compute the LDOS for 1/8th of the domain.

nx, ny = 20, 20
X = np.linspace(0, 8, nx)
Y = np.linspace(0, 8, ny)
ldos = np.zeros((nx, ny))

for j, y in enumerate(Y):
    for i, x in enumerate(X):
        ldos[i, j] = s.local_density_of_states(x, y) if j <= i else ldos[j, i]

##############################################################################
# Rearrange the map and visualize it.

X = np.linspace(-8, 8, 2 * nx - 1)
Y = np.linspace(-8, 8, 2 * ny - 1)
LX = np.vstack([np.flipud(ldos[1:, :]), ldos])
LDOS = np.hstack([np.fliplr(LX[:, 1:]), LX])

v = np.log10(LDOS * gy.pi * gy.c**2 / (2 * ls.pulsation))

fig, ax = plt.subplots(figsize=(2.6, 2.2))
cs = plt.contourf(X, Y, v, cmap="Spectral_r", levels=31)
plot_rods(ax, rod_positions)
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.axis("square")
plt.xlabel(r"$x/d$")
plt.ylabel(r"$y/d$")
plt.xticks(np.arange(-8, 9, 4))
plt.yticks(np.arange(-8, 9, 4))
plt.colorbar(cs, fraction=0.04, pad=0.08)
plt.title(r"$\log_{10}(\rho \pi c^2/2\omega)$")
plt.tight_layout()
