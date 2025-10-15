#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.3
# License: MIT
# See the documentation at gyptis.gitlab.io

"""
Band diagram of 2D photonic crystal
===================================

Calculation of the band diagram of a two-dimensional photonic crystal.
"""


# sphinx_gallery_thumbnail_number = -1


import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy
from gyptis import pi
from gyptis.utils import bands

##############################################################################
# Reference results are taken from  :cite:p:`Joannopoulos2008` (Chapter 5 Fig. 2).
#
# The structure is a square lattice of dielectric
# columns, with radius r and dielectric constant :math:`\varepsilon`.
# The material is invariant along the z direction  and periodic along
# :math:`x` and :math:`y` with lattice constant :math:`a`.
# We will define the geometry using the class :class:`~gyptis.Lattice`,
# defining the two vectors of periodicity:

a = 1  # unit cell size
vectors = (a, 0), (0, a)  # vectors defining the unit cell
R = 0.2 * a  # inclusion radius

lattice = gy.Lattice(dim=2, vectors=vectors)

##############################################################################
# Next, we add a cylinder and compute the boolean fragments

circ = lattice.add_circle(a / 2, a / 2, 0, R)
circ, cell = lattice.fragment(circ, lattice.cell)

##############################################################################
# One needs to define physical domains associated with the basic geometrical
# entities:

lattice.add_physical(cell, "background")
_ = lattice.add_physical(circ, "inclusion")

##############################################################################
# Set minimum mesh size

lattice.set_size("background", a / 10)
lattice.set_size("inclusion", R / 10)

##############################################################################
# Finally, we can build the geometry, which will also construct the mesh.

lattice.build()

##############################################################################
# Material parameters are defined with a python dictionary:

epsilon = dict(background=1, inclusion=8.9)
mu = dict(background=1, inclusion=1)

##############################################################################
# We can now instanciate the simulation class :class:`~gyptis.PhotonicCrystal`.
# We will compute eigenpairs at the :math:`X` point of the Brillouin zone, *i.e.*
# the propagation vector is :math:`\mathbf k = (\pi/a,0)`.

phc = gy.PhotonicCrystal(
    lattice,
    epsilon,
    mu,
    propagation_vector=(pi / a, 0),
    polarization="TE",
    degree=2,
)


##############################################################################
# To calculate the eigenvalues and eigenvectors, we call the
# :meth:`~gyptis.PhotonicCrystal.eigensolve` method.

solution = phc.eigensolve(n_eig=6, target=0.5)


##############################################################################
# The results can be accessed through the `phc.solution` attribute
# (a dictionary).

ev_norma = phc.solution["eigenvalues"].real * a / (2 * pi)
print("Normalized eigenfrequencies")
print("---------------------------")
print(ev_norma)

##############################################################################
# Lets plot the field map of the modes.

eig_vects = phc.solution["eigenvectors"]
for mode, eval in zip(eig_vects, ev_norma):
    gy.plot(mode.real, cmap="gyptis")
    plt.title(rf"$\omega a/2\pi c = {eval:0.3f}$")
    H = phc.formulation.get_dual(mode, 1)
    gy.dolfin.plot(H.real, cmap="Greys")
    lattice.plot_subdomains()
    plt.axis("off")


##############################################################################
# We define here the wavevector path:

nband = 21
Gamma = 0, 0
X = pi, 0
M = pi, pi
sym_points = Gamma, X, M, Gamma
ks = bands.init_bands(sym_points, nband)

##############################################################################
# Calculate the band diagram:

band_diag = {}
for polarization in ["TE", "TM"]:
    evs = []
    for k in ks:
        phc = gy.PhotonicCrystal(
            lattice,
            epsilon,
            mu,
            propagation_vector=k,
            polarization=polarization,
            degree=1,
        )
        phc.eigensolve(n_eig=6, target=0.1)
        ev_norma = phc.solution["eigenvalues"].real * a / (2 * pi)
        evs.append(ev_norma)
    band_diag[polarization] = evs


##############################################################################
# Plot the bands:

klabels = [r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$"]


plt.figure(figsize=(3.2, 2.5))
plotTM = bands.plot_bands(
    sym_points,
    nband,
    band_diag["TM"],
    ls="-",
    marker="",
    xtickslabels=klabels,
    color="#4199b0",
)
plotTE = bands.plot_bands(
    sym_points,
    nband,
    band_diag["TE"],
    ls="-",
    marker="",
    xtickslabels=klabels,
    color="#cf5268",
)
plt.annotate("TM modes", (1, 0.05), c="#4199b0")
plt.annotate("TE modes", (0.33, 0.33), c="#cf5268")
plt.ylim(0, 0.8)
plt.ylabel(r"Frequency $\omega a/2\pi c$")
plt.tight_layout()
