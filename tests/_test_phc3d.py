#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

import pytest

from gyptis import dolfin
from gyptis.phc3d import *
from gyptis.plot import *

dolfin.set_log_level(10)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 5
dolfin.parameters["ghost_mode"] = "shared_facet"

a = 1
v = (a, 0, 0), (0, a, 0), (0, 0, a)
# v = (a,0, 0), (0, a,0), (1.2*a,1.5*a,a*0.7)
# v = (0.21*a,0.24*a,a*0.3), (0.3*a,0.1*a,a*0.1), (0.2*a,0.4*a,a*0.12)
R = 0.25 * a
n_eig = 6

lattice = Lattice3D(v)
sphere = lattice.add_sphere(a / 2, a / 2, a / 2, R)
sphere, cell = lattice.fragment(sphere, lattice.cell)
lattice.add_physical(cell, "background")
lattice.add_physical(sphere, "inclusion")
periodic_id = lattice.get_periodic_bnds()

for k, v in periodic_id.items():
    lattice.add_physical(v, k, 2)

lattice.set_size("background", 0.1)
lattice.set_size("inclusion", 0.1)

lattice.build()

bcs = {k: "PEC" for k, v in periodic_id.items()}
# pbc = Periodic3D(lattice)

eps_inclusion = 1
epsilon = dict(background=1, inclusion=eps_inclusion)
mu = dict(background=1, inclusion=1)

phc = PhotonicCrystal3D(
    lattice,
    epsilon,
    mu,
    propagation_vector=(0, 0, 0),
    degree=2,
    boundary_conditions=bcs,
)
phc.eigensolve(n_eig=12, wavevector_target=1)
ev_norma = np.array(phc.solution["eigenvalues"]) * a / (np.pi)
ev = np.array(phc.solution["eigenvalues"])

true_eig = (
    np.pi
    / a
    * np.sort(
        np.array(
            [
                (m**2 + n**2 + p**2) ** 0.5
                for m in range(6)
                for n in range(6)
                for p in range(6)
            ]
        )
    )
)
true_eig_norma = true_eig * a / (np.pi)

# print(ev_norma)
# print(true_eig_norma)

print(ev_norma**2)
print(true_eig_norma**2)

#
# @pytest.mark.parametrize(
#     "degree,polarization", [(1, "TM"), (2, "TM"), (1, "TE"), (2, "TE")]
# )
# def test_phc(degree, polarization):
#
#     phc = PhotonicCrystal2D(
#         lattice,
#         epsilon,
#         mu,
#         propagation_vector=(0.1 * np.pi / a, 0.2 * np.pi / a),
#         polarization=polarization,
#         degree=degree,
#     )
#     phc.eigensolve(n_eig=6, wavevector_target=0.1)
#     ev_norma = np.array(phc.solution["eigenvalues"]) * a / (2 * np.pi)
#     ev_norma = ev_norma[:n_eig].real
#
#     eig_vects = phc.solution["eigenvectors"]
#     mode, eval = eig_vects[4], ev_norma[4]
#     fplot = project(mode.real, phc.formulation.real_function_space)
#     dolfin.plot(fplot, cmap="RdBu_r")
