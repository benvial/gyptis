#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.0
# License: MIT
# See the documentation at gyptis.gitlab.io

"""
Band diagram of 3D photonic crystal
===================================

Face-centered cubic (fcc) lattice of close-packed dielectric spheres .
"""


# sphinx_gallery_thumbnail_number = -1


import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy
from gyptis import pi
from gyptis.utils import bands

gy.set_log_level("CRITICAL", 0)

gy.dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

##############################################################################
# Reference results are taken from  :cite:p:`Joannopoulos2008` (Chapter 6 Fig. 2).
#
pmesh = 1

a = 1  # unit cell size
# vectors = (
#     (a / 2, a / 2, 0),
#     (0, a / 2, a / 2),
#     (a / 2, 0, a / 2),
# )  # vectors defining the unit cell
vectors = (
    (a, 0, 0),
    (0, a, 0),
    (0, 0, a),
)  # vectors defining the unit cell
R = 0.2 * a  # inclusion radius

vectors = np.array(vectors)

lattice = gy.Lattice(dim=3, vectors=vectors, periodic_tol=1e-8)

##############################################################################
# Next, we add a cylinder and compute the boolean fragments
# spheres = []
# positions = (
#     (0, 0, 0),
#     vectors[0],
#     vectors[1],
#     vectors[2],
#     vectors[0] + vectors[1],
#     vectors[0] + vectors[2],
#     vectors[1] + vectors[2],
#     vectors[0] + vectors[1]+ vectors[2],
# )
# for pos in positions:
#     sphere = lattice.add_sphere(*pos, R)
#     spheres.append(sphere)
# spheres = lattice.intersect(lattice.cell,spheres,removeObject=False)
# *spheres,cell = lattice.fragment(lattice.cell,spheres)

R = 0.2 * a
spheres = lattice.add_sphere(a / 2, a / 2, a / 2, R)
*spheres, cell = lattice.fragment(lattice.cell, spheres)

# lattice.remove_all_duplicates()

##############################################################################
# One needs to define physical domains associated with the basic geometrical
# entities:

lattice.add_physical(cell, "background")
_ = lattice.add_physical(spheres, "inclusion")

##############################################################################
# Set minimum mesh size

lattice.set_size("background", a / pmesh)
lattice.set_size("inclusion", R / pmesh)

##############################################################################
# Finally, we can build the geometry, which will also construct the mesh.

lattice.build(0)


##############################################################################
# Material parameters are defined with a python dictionary:

epsilon = dict(background=1, inclusion=13 - 8j)
mu = dict(background=1, inclusion=1)

##############################################################################
# We can now instanciate the simulation class :class:`~gyptis.PhotonicCrystal`.
# We will compute eigenpairs at the :math:`X` point of the Brillouin zone, *i.e.*
# the propagation vector is :math:`\mathbf k = (\pi/a,0)`.

phc = gy.models.PhotonicCrystal3D(
    lattice,
    epsilon,
    mu,
    propagation_vector=(1, 1, 1),
    degree=2,
)


##############################################################################
# To calculate the eigenvalues and eigenvectors, we call the
# :meth:`~gyptis.PhotonicCrystal.eigensolve` method.

from gyptis.utils.time import list_time

solution = phc.eigensolve(n_eig=6, target=2.1, tol=1e-6)
# list_time()

##############################################################################
# The results can be accessed through the `phc.solution` attribute
# (a dictionary).


ev_norma = phc.solution["eigenvalues"]  # .real * a / (2 * pi)
print("Normalized eigenfrequencies")
print("---------------------------")
print(ev_norma)


##############################################################################
# We define here the wavevector path:

nband = 11
Gamma = 0, 0
X = pi, 0
M = pi, pi
sym_points = Gamma, X, M, Gamma
ks = bands.init_bands(sym_points, nband)

##############################################################################
# Calculate the band diagram:

band_diag = []
for kx, ky in ks:
    print(kx, ky)
    phc = gy.models.PhotonicCrystal3D(
        lattice,
        epsilon,
        mu,
        propagation_vector=(kx, ky, 0),
        degree=1,
    )
    phc.eigensolve(n_eig=6, target=0.1)
    ev_norma = phc.solution["eigenvalues"].real * a / (2 * pi)
    band_diag.append(ev_norma)


##############################################################################
# Plot the bands:

klabels = [r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$"]


plt.figure(figsize=(3.2, 2.5))
bands.plot_bands(
    sym_points,
    nband,
    band_diag,
    ls="-",
    marker="",
    xtickslabels=klabels,
    color="#1c872c",
)
# plt.ylim(0, 0.8)
plt.ylabel(r"Frequency $\omega a/2\pi c$")
plt.tight_layout()
plt.show()
