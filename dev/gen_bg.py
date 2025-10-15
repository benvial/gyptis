#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.3
# License: MIT
# See the documentation at gyptis.gitlab.io

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy

plt.ion()
plt.close("all")

wavelength = 3.5
pmesh = 5  # mesh parameter
a = 0.25  # radius of the rods
n_cyl = 3  # index of the rods

Nx, Ny = 10, 4

dc = 0.0
dr = 0.0

rod_positions = [
    [(1 - dc * np.random.rand()) * i, (1 - dc * np.random.rand()) * j]
    for i in range(-Nx, Nx + 1)
    for j in range(-Ny, Ny + 1)
]

rod_radii = [
    (1 - dr * np.random.rand()) * a
    for i in range(-Nx, Nx + 1)
    for j in range(-Ny, Ny + 1)
]


def plot_rods(ax, rod_positions, rod_radii, *args, **kwargs):
    for pos, R in zip(rod_positions, rod_radii):
        circle = plt.Circle(pos, R, fill=False, *args, **kwargs)
        ax.add_patch(circle)


def create_geometry(wavelength, pml_width, group=False):
    lmin = wavelength / pmesh
    geom = gy.BoxPML(
        dim=2,
        box_size=(2 * Nx + 1 + 2 * wavelength, 2 * Ny + 1 + 2 * wavelength),
        pml_width=(pml_width, pml_width),
    )
    box = geom.box
    cylinders = []
    for pos, R in zip(rod_positions, rod_radii):
        cyl = geom.add_circle(*pos, 0, R)
        cylinders.append(cyl)
    *cylinders, box = geom.fragment(cylinders, box)
    geom.add_physical(box, "box")
    [geom.set_size(pml, lmin * 1) for pml in geom.pmls]
    geom.set_size("box", lmin)
    if group:
        geom.add_physical(cylinders, f"cylinders")
        geom.set_size(f"cylinders", lmin / n_cyl)
    else:
        # we could define physical domains for each rod but that is slower
        # when assembling and solving the sctattering problem
        for i, cyl in enumerate(cylinders):
            geom.add_physical(cyl, f"cylinder_{i}")
            geom.set_size(f"cylinder_{i}", lmin / n_cyl)
    geom.build()
    return geom


geom = create_geometry(wavelength, pml_width=wavelength, group=True)
geom.plot_mesh(lw=0.2)
# geom.plot_subdomains(lw=0.8,c="r")
plot_rods(plt.gca(), rod_positions, rod_radii, lw=0.3)


ls = gy.LineSource(
    wavelength=wavelength, position=(-Nx - 1, 0), domain=geom.mesh, degree=2
)

epsilon = {d: n_cyl**2 for d in geom.domains}
epsilon["box"] = 1
mu = {d: 1 for d in geom.domains}

##############################################################################
# Instanciate and solve the scattering problem:

s = gy.Scattering(geom, epsilon, mu, ls, degree=2, polarization="TM")
s.solve()
G = s.solution["total"]


# G = s.solution["diffracted"]

##############################################################################
# Here we project the function to plot on a suitable function space using an
# iterative solver:

v = gy.dolfin.ln(abs(G)) / gy.dolfin.ln(10)


v = G.imag
vplot = gy.project(
    v,
    s.formulation.real_function_space,
    solver_type="cg",
    preconditioner_type="jacobi",
)

##############################################################################
# Plot the Green's function:


from matplotlib import colors

cols = [
    "#b1a9a9",
    "#45957d",
    "#e2e2e2",
    "#d25f5f",
    "#453e3e",
    "#333333",
    "#d2c35f",
]
cmap = colors.ListedColormap(cols)

fig, ax = plt.subplots(figsize=(5, 3))
cs = gy.dolfin.plot(vplot, mode="contourf", cmap=cmap, levels=7 * 3)
plot_rods(ax, rod_positions, rod_radii, lw=0.3)
plt.axis("off")
plt.tight_layout()
plt.xlim(-Nx - 2, Nx + 2)
plt.ylim(-Ny - 2, Ny + 2)
plt.savefig(
    "bg.svg",
    bbox_inches="tight",
    pad_inches=0,
)
