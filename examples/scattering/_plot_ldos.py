#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.1
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

gy.dolfin.parameters["form_compiler"]["quadrature_degree"] = 8

##############################################################################
# Reference results are taken from :cite:p:`Asatryan2001`.

wavelength = 3.5
pmesh = 5  # mesh parameter
a = 0.3  # radius of the rods
n_cyl = 3  # index of the rods


def plot_rods(ax, rod_positions, radius):
    for pos in rod_positions:
        circle = plt.Circle(pos, radius, fill=False)
        ax.add_patch(circle)


##############################################################################
# We define the position of the centers of the rods:

rod_positions = [(i, j) for i in range(-3, 4) for j in range(-3, 4)]
rod_positions += [(i, j) for i in [-4, 4] for j in range(-3, 4)]
rod_positions += [(i, j) for j in [-4, 4] for i in range(-3, 4)]
rod_positions += [(i, 0) for i in [-5, 5]]
rod_positions += [(0, i) for i in [-5, 5]]


##############################################################################
# Build and mesh the geometry:


def create_geometry(rod_positions, wavelength, pml_width, group=False):
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


# geom = create_geometry(rod_positions,wavelength, pml_width=wavelength, group=True)

# ##############################################################################
# # Define the line excitation and materials

# ls = gy.LineSource(wavelength=wavelength, position=(0, 7.3), domain=geom.mesh, degree=2)

# epsilon = {d: n_cyl**2 for d in geom.domains}
# epsilon["box"] = 1

# ##############################################################################
# # Instanciate and solve the scattering problem:

# s = gy.Scattering(geom, epsilon, source=ls, degree=2, polarization="TM")
# s.solve()
# G = s.solution["total"]

# ##############################################################################
# # Here we project the function to plot on a suitable function space using an
# # iterative solver:

# v = gy.dolfin.ln(abs(G)) / gy.dolfin.ln(10)
# vplot = gy.project(
#     v,
#     s.formulation.real_function_space,
#     solver_type="cg",
#     preconditioner_type="jacobi",
# )

# ##############################################################################
# # Plot the Green's function:

# fig, ax = plt.subplots(figsize=(2.6, 2.2))
# cs = gy.dolfin.plot(vplot, mode="contourf", cmap="Spectral_r", levels=31)
# plot_rods(ax, rod_positions,a)
# plt.axis("square")
# plt.xlabel(r"$x/d$")
# plt.ylabel(r"$y/d$")
# plt.xlim(-8, 8)
# plt.ylim(-8, 8)
# plt.xticks(np.arange(-8, 9, 4))
# plt.yticks(np.arange(-8, 9, 4))
# plt.colorbar(cs, fraction=0.04, pad=0.08)
# plt.title(r"$\log_{10}|G\,|$")
# plt.tight_layout()


# ##############################################################################
# # Due to symmetry we will only compute the LDOS for 1/8th of the domain.

# nx, ny = 20, 20
# X = np.linspace(0, 8, nx)
# Y = np.linspace(0, 8, ny)
# ldos = np.zeros((nx, ny))

# for j, y in enumerate(Y):
#     for i, x in enumerate(X):
#         ldos[i, j] = s.local_density_of_states(x, y) if j <= i else ldos[j, i]

# ##############################################################################
# # Rearrange the map and visualize it.

# X = np.linspace(-8, 8, 2 * nx - 1)
# Y = np.linspace(-8, 8, 2 * ny - 1)
# LX = np.vstack([np.flipud(ldos[1:, :]), ldos])
# LDOS = np.hstack([np.fliplr(LX[:, 1:]), LX])

# v = np.log10(LDOS * gy.pi * gy.c**2 / (2 * ls.pulsation))

# fig, ax = plt.subplots(figsize=(2.6, 2.2))
# cs = plt.contourf(X, Y, v, cmap="Spectral_r", levels=31)
# plot_rods(ax, rod_positions)
# plt.xlim(-8, 8)
# plt.ylim(-8, 8)
# plt.axis("square")
# plt.xlabel(r"$x/d$")
# plt.ylabel(r"$y/d$")
# plt.xticks(np.arange(-8, 9, 4))
# plt.yticks(np.arange(-8, 9, 4))
# plt.colorbar(cs, fraction=0.04, pad=0.08)
# plt.title(r"$\log_{10}(\rho \pi c^2/2\omega)$")
# plt.tight_layout()


##############################################################################
# For TE polarization, it is possible to form a band gap with air cylinders
# in a dense, homogeneous matrix and to generate a full band gap with
# a hexagonal array.


plt.close("all")
plt.ion()

wavelength = 2.25
n_bg = 13**0.5
radius = 0.38
# radius = (0.48 / np.pi) ** 0.5

rod_positions = []
for i in range(-4, 5):
    I = 9 - abs(i)
    for j in range(0, I):
        posx = i * 3**0.5 / 2
        posy = j - I / 2 + 0.5
        pos = posx, posy
        rod_positions.append(pos)


geom = create_geometry(rod_positions, wavelength, pml_width=wavelength, group=True)

epsilon = {d: 1 for d in geom.domains}
epsilon["box"] = n_bg**2

ls = gy.LineSource(wavelength=wavelength, position=(0, 0), domain=geom.mesh, degree=2)
dp = gy.Dipole(
    wavelength=wavelength,
    position=(0, 0),
    domain=geom.mesh,
    degree=2,
)

s = gy.Scattering(geom, epsilon, source=ls, degree=2, polarization="TE")


s.solve()
G = s.solution["total"]

v = gy.dolfin.ln(abs(G)) / gy.dolfin.ln(10)
vplot = gy.project_iterative(
    v,
    s.formulation.real_function_space,
)

fig, ax = plt.subplots()
cs = gy.dolfin.plot(vplot, mode="contourf", cmap="Spectral_r", levels=31)
plot_rods(ax, rod_positions, radius)
plt.axis("square")
plt.xlabel(r"$x/d$")
plt.ylabel(r"$y/d$")
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.xticks(np.arange(-8, 9, 4))
plt.yticks(np.arange(-8, 9, 4))
plt.colorbar(cs, fraction=0.04, pad=0.08)
plt.title(r"$\log_{10}|G\,|$")
plt.plot(*ls.position, "xk")
plt.tight_layout()


##############################################################################
# Due to symmetry we will only compute the LDOS for 1/8th of the domain.

nx, ny = 20, 20
X = np.linspace(0, 8, nx)
Y = np.linspace(0, 8, ny)
ldos = np.zeros((nx, ny))


from gyptis import Dipole, LineSource, c, dolfin

self = s
x, y = 1, 1


# if True:
def _local_density_of_states_TE(self, x, y):
    # greens_tensor = np.zeros((2,2), dtype=complex)
    trace_greens_tensor = 0
    for comp in [0, 1]:
        self.source = Dipole(
            wavelength=self.source.wavelength,
            position=(x, y),
            domain=self.mesh,
            degree=self.degree,
        )
        if hasattr(self, "solution"):
            self.assemble_rhs()
            self.solve_system(again=True)
        else:
            self.solve()
        u = self.solution["total"]
        eps = dolfin.DOLFIN_EPS_LARGE
        delta = 1 + eps
        evalpoint = x * delta, y * delta
        if evalpoint[0] == 0:
            evalpoint = eps, evalpoint[1]
        if evalpoint[1] == 0:
            evalpoint = evalpoint[0], eps
        print("solved")
        dual = self.formulation.get_dual(u)
        V = dolfin.FunctionSpace(self.mesh, "CG", self.formulation.degree - 1)
        # V = dolfin.FunctionSpace(self.mesh, "DG", 0)
        # V = self.formulation.real_function_space
        v = gy.project_iterative(dual[comp].imag, V)
        val = v(evalpoint)
        trace_greens_tensor += val
    # print(v(evalpoint))
    out = -2 * self.source.pulsation / (np.pi * c**2) * trace_greens_tensor
    print(out)
    return out


for j, y in enumerate(Y):
    for i, x in enumerate(X):
        ldos[i, j] = _local_density_of_states_TE(s, x, y) if j <= i else ldos[j, i]

##############################################################################
# Rearrange the map and visualize it.

X = np.linspace(-8, 8, 2 * nx - 1)
Y = np.linspace(-8, 8, 2 * ny - 1)
LX = np.vstack([np.flipud(ldos[1:, :]), ldos])
LDOS = np.hstack([np.fliplr(LX[:, 1:]), LX])

v = np.log10((LDOS) * gy.pi * gy.c**2 / (2 * ls.pulsation * n_bg**2))

# fig, ax = plt.subplots(figsize=(2.6, 2.2))
fig, ax = plt.subplots()
cs = plt.contourf(X, Y, v, cmap="Spectral_r", levels=31)
plot_rods(ax, rod_positions, radius)
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
