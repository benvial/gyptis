#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import pytest

from gyptis.homogenization import *

a = 1
v = (a, 0, 0), (0, a, 0), (0, 0, a)
Rx = 0.1 * a
Ry = 0.2 * a
Rz = 0.3 * a
lmin = 0.1

lattice = Lattice3D(v)
circ = lattice.add_sphere(a / 2, a / 2, a / 2, Rx)
circ, cell = lattice.fragment(circ, lattice.cell)
lattice.dilate(lattice.dimtag(circ), a / 2, a / 2, a / 2, 1, Ry / Rx, Rz / Rx)
lattice.add_physical(cell, "background")
lattice.add_physical(circ, "inclusion")
lattice.set_size("background", lmin)
lattice.set_size("inclusion", lmin)
lattice.build()
# lattice.build(1,read_mesh=0,read_info=0, generate_mesh=1,periodic=0)


eps_incl = 4 - 3j
degree = 2
epsilon = dict(background=1.25, inclusion=eps_incl)
mu = dict(background=1, inclusion=1)

hom = Homogenization3D(
    lattice,
    epsilon,
    mu,
    degree=degree,
)
eps_eff = hom.get_effective_permittivity()
