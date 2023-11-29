#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

import pytest

from gyptis.models import *

a = 1
v = (a, 0), (0, a)
R1 = 0.3 * a
R2 = 0.4 * a
lmin = 0.1

lattice = Lattice2D(v)
circ = lattice.add_ellipse(a / 2, a / 2, 0, R1, R2)
circ, cell = lattice.fragment(circ, lattice.cell)
lattice.add_physical(cell, "background")
lattice.add_physical(circ, "inclusion")
lattice.set_size("background", lmin)
lattice.set_size("inclusion", lmin)
lattice.build()


@pytest.mark.parametrize(
    "degree,epsincl,muincl",
    [(1, 4 - 3j, 1), (2, 4 - 3j, 1), (1, 4 - 3j, 3 - 0.1j), (2, 4 - 3j, 3 - 0.1j)],
)
def test_hom(degree, epsincl, muincl):
    EPS = []
    MU = []
    for eps_inclusion, mu_inclusion in zip(
        [epsincl, epsincl * np.eye(3)], [muincl, muincl * np.eye(3)]
    ):
        epsilon = dict(background=1.25, inclusion=eps_inclusion)
        mu = dict(background=1, inclusion=mu_inclusion)

        hom = Homogenization2D(
            lattice,
            epsilon,
            mu,
            degree=degree,
        )
        eps_eff = hom.get_effective_permittivity()
        EPS.append(eps_eff)
        mu_eff = hom.get_effective_permeability()
        MU.append(mu_eff)

    assert np.allclose(*EPS)
    assert np.allclose(*MU)
