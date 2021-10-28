#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import pytest

from gyptis.homogenization import *

a = 1
v = (a, 0), (0, a)
R1 = 0.3 * a
R2 = 0.4 * a
lmin = 0.05

lattice = Lattice2D(v)
circ = lattice.add_ellipse(a / 2, a / 2, 0, R1, R2)
circ, cell = lattice.fragment(circ, lattice.cell)
lattice.add_physical(cell, "background")
lattice.add_physical(circ, "inclusion")
lattice.set_size("background", lmin)
lattice.set_size("inclusion", lmin)
lattice.build()


@pytest.mark.parametrize(
    "degree,polarization", [(1, "TM"), (2, "TM"), (1, "TE"), (2, "TE")]
)
def test_hom(degree, polarization):
    EPS = []
    for eps_incl in [4 - 3j, (4 - 3j) * np.eye(3)]:

        epsilon = dict(background=1.25, inclusion=eps_incl)
        mu = dict(background=1, inclusion=1)

        hom = Homogenization2D(
            lattice,
            epsilon,
            mu,
            polarization=polarization,
            degree=degree,
        )
        eps_eff = hom.get_effective_permittivity()
        EPS.append(eps_eff)

    assert np.allclose(*EPS)


# plt.ion()
# fig,ax=plt.subplots(1,2,figsize=(5,2))
# plot(hom.solution["x"].real,geometry=lattice,ax=ax[0])
# plot(hom.solution["y"].real,geometry=lattice,ax=ax[1])
# [a.set_axis_off()for a in ax]
# ax[0].set_title("$V_x$")
# ax[1].set_title("$V_y$")
# plt.tight_layout()
