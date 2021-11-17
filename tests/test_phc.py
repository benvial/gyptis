#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import pytest

from gyptis.photonic_crystal import *
from gyptis.plot import *

# dy = 0.5 * 2 ** 0.5
# v = (1, 0), (dy, dy)
# v = (1, 0), (0, 1)
# R = 0.3
#
# lattice = Lattice2D(v)
#
# circ = lattice.add_circle(0, 0, 0, R)
# circ_0, circ1, cell = lattice.fragment(circ, lattice.cell)
# lattice.remove([(2, circ1)], recursive=True)
#
# circ = lattice.add_circle(v[1][0], v[1][1], 0, R)
# circ1, circ_1, cell = lattice.fragment(circ, cell)
# lattice.remove([(2, circ1)], recursive=True)
#
# circ = lattice.add_circle(v[0][0], v[0][1], 0, R)
# circ1, circ_2, cell = lattice.fragment(circ, cell)
# lattice.remove([(2, circ1)], recursive=True)
#
# circ = lattice.add_circle(v[0][0] + v[1][0], v[1][1], 0, R)
# circ1, circ_3, cell = lattice.fragment(circ, cell)
# lattice.remove([(2, circ1)], recursive=True)
#
# lattice.add_physical(cell, "background")
# lattice.add_physical([circ_0, circ_1, circ_2, circ_3], "inclusion")
#
# print(lattice.get_periodic_bnds())
# # l.set_mesh_size({"background": 0.1, "inclusion": 0.03})
# lattice.set_size("background", 0.1)
# lattice.set_size("inclusion", 0.1)
#
# lattice.build()


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


@pytest.mark.parametrize(
    "degree,polarization", [(1, "TM"), (2, "TM"), (1, "TE"), (2, "TE")]
)
def test_phc(degree, polarization):

    phc = PhotonicCrystal2D(
        lattice,
        epsilon,
        mu,
        propagation_vector=(0.1 * np.pi / a, 0.2 * np.pi / a),
        polarization=polarization,
        degree=degree,
    )
    phc.eigensolve(n_eig=6, wavevector_target=0.1)
    ev_norma = np.array(phc.solution["eigenvalues"]) * a / (2 * np.pi)
    ev_norma = ev_norma[:n_eig].real

    eig_vects = phc.solution["eigenvectors"]
    mode, eval = eig_vects[4], ev_norma[4]
    fplot = project(mode.real, phc.formulation.real_function_space)
    dolfin.plot(fplot, cmap="RdBu_r")
