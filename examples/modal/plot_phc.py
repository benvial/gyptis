#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

"""
Band diagram of 2D photonic crystal
===================================

Calculation of the band diagram of a two-dimensional photonic crystal.
"""


# sphinx_gallery_thumbnail_number = -1

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

from gyptis import Lattice, PhotonicCrystal
from gyptis.complex import project
from gyptis.plot import *

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

lattice = Lattice(dim=2, vectors=vectors)

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

phc = PhotonicCrystal(
    lattice,
    epsilon,
    mu,
    propagation_vector=(np.pi / a, 0),
    polarization="TE",
    degree=2,
)


##############################################################################
# To calculate the eigenvalues and eigenvectors, we call the
# :meth:`~gyptis.PhotonicCrystal.eigensolve` method.

solution = phc.eigensolve(n_eig=6, wavevector_target=0.5)


##############################################################################
# The results can be accessed through the `phc.solution` attribute
# (a dictionary).

ev_norma = phc.solution["eigenvalues"].real * a / (2 * np.pi)
print("Normalized eigenfrequencies")
print("---------------------------")
print(ev_norma)

##############################################################################
# Lets plot the field map of the modes.

eig_vects = phc.solution["eigenvectors"]
for mode, eval in zip(eig_vects, ev_norma):
    plot(mode.real, cmap="gyptis")
    plt.title(fr"$\omega a/2\pi c = {eval:0.3f}$")
    H = phc.formulation.get_dual(mode, 1)
    dolfin.plot(H.real, cmap="Greys")
    lattice.plot_subdomains()
    plt.axis("off")


##############################################################################
# We define here the wavevector path:

Nb = 21
K = np.linspace(0, np.pi / a, Nb)
bands = np.zeros((3 * Nb - 3, 2))
bands[:Nb, 0] = K
bands[Nb : 2 * Nb, 0] = K[-1]
bands[Nb : 2 * Nb - 1, 1] = K[1:]
bands[2 * Nb - 1 : 3 * Nb - 3, 0] = bands[2 * Nb - 1 : 3 * Nb - 3, 1] = np.flipud(K)[
    1:-1
]

##############################################################################
# Calculate the band diagram:

BD = {}
for polarization in ["TE", "TM"]:
    ev_band = []
    for kx, ky in bands:
        phc = PhotonicCrystal(
            lattice,
            epsilon,
            mu,
            propagation_vector=(kx, ky),
            polarization=polarization,
            degree=1,
        )
        phc.eigensolve(n_eig=6, wavevector_target=0.1)
        ev_norma = phc.solution["eigenvalues"].real * a / (2 * np.pi)
        ev_band.append(ev_norma)
    # append first value since this is the same point
    ev_band.append(ev_band[0])
    BD[polarization] = ev_band


##############################################################################
# Plot the bands:

bands_plot = np.zeros(3 * Nb - 2)
bands_plot[:Nb] = K
bands_plot[Nb : 2 * Nb - 1] = K[-1] + K[1:]
bands_plot[2 * Nb - 1 : 3 * Nb - 2] = 2 * K[-1] + 2 ** 0.5 * K[1:]


plt.figure(figsize=(3.2, 2.5))
plotTM = plt.plot(bands_plot, BD["TM"], c="#4199b0")
plotTE = plt.plot(bands_plot, BD["TE"], c="#cf5268")
plt.annotate("TM modes", (1, 0.05), c="#4199b0")
plt.annotate("TE modes", (0.33, 0.33), c="#cf5268")
plt.ylim(0, 0.8)
plt.xlim(0, bands_plot[-1])
plt.xticks(
    [0, K[-1], 2 * K[-1], bands_plot[-1]], ["$\Gamma$", "$X$", "$M$", "$\Gamma$"]
)
plt.axvline(K[-1], c="k", lw=0.3)
plt.axvline(2 * K[-1], c="k", lw=0.3)
plt.ylabel(r"Frequency $\omega a/2\pi c$")
plt.tight_layout()
