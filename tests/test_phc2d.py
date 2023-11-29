#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

import numpy as np
import pytest


@pytest.mark.parametrize(
    "degree,polarization", [(1, "TM"), (2, "TM"), (1, "TE"), (2, "TE")]
)
def test_phc(degree, polarization):
    from gyptis import dolfin, project
    from gyptis.models.phc2d import Lattice2D, PhotonicCrystal2D

    a = 1
    v = (a, 0), (0, a)
    R = 0.2 * a
    n_eig = 6

    lattice = Lattice2D(v)
    circ = lattice.add_circle(a / 2, a / 2, 0, R)
    circ, cell = lattice.fragment(circ, lattice.cell)
    lattice.add_physical(cell, "background")
    lattice.add_physical(circ, "inclusion")

    lattice.set_size("background", 0.05)
    lattice.set_size("inclusion", 0.05)

    lattice.build()
    eps_inclusion = 8.9
    epsilon = dict(background=1, inclusion=eps_inclusion)
    mu = dict(background=1, inclusion=1)

    phc = PhotonicCrystal2D(
        lattice,
        epsilon,
        mu,
        propagation_vector=(0.1 * np.pi / a, 0.2 * np.pi / a),
        polarization=polarization,
        degree=degree,
    )
    phc.eigensolve(n_eig=n_eig, wavevector_target=0.1)
    eig_vects = phc.solution["eigenvectors"]
    mode = eig_vects[4]
    fplot = project(mode.real, phc.formulation.real_function_space)
    dolfin.plot(fplot, cmap="RdBu_r")
