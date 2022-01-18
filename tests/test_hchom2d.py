#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import copy

import pytest

from gyptis.homogenization import *

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
# gmsh.write("test.geo_unrolled")
# lattice.build(1,finalize=False)
gmsh.finalize()

gmsh.initialize()
gmsh.clear()
# gmsh.model.add("zob")
gmsh.open("test.geo_unrolled")
lattice_bg = copy.copy(lattice)
idphys = lattice_bg.subdomains["surfaces"]["inclusion"]
id = gmsh.model.get_entities_for_physical_group(2, idphys)
lattice_bg.remove(lattice_bg.dimtag(id))
lattice_bg.build(1, finalize=False)

xsx
circ = lattice.add_ellipse(a / 2, a / 2, 0, R1, R2)
circ, cell = lattice.fragment(circ, lattice.cell)
lattice.add_physical(cell, "background")
lattice.add_physical(circ, "inclusion")
lattice.set_size("background", lmin)
lattice.set_size("inclusion", lmin)
lattice.build(1, finalize=False)
gmsh.write("test.geo_unrolled")


lattice_bg = Lattice2D(v)
circ = lattice_bg.add_ellipse(a / 2, a / 2, 0, R1, R2)
circ, cell = lattice_bg.fragment(circ, lattice_bg.cell)
lattice_bg.remove(lattice_bg.dimtag(circ))
lattice_bg.add_physical(cell, "background")
lattice_bg.set_size("background", lmin)
lattice_bg.build(1)

lattice_incl = Lattice2D(v)
circ = lattice_incl.add_ellipse(a / 2, a / 2, 0, R1, R2)
circ, cell = lattice_incl.fragment(circ, lattice_incl.cell)
lattice_incl.remove(lattice_incl.dimtag(cell))
lattice_incl.add_physical(circ, "inclusion")
lattice_incl.set_size("inclusion", lmin)
lattice_incl.build(1)
import gmsh

# import copy
# l1 = copy.copy(lattice)
#
#
# # gmsh.model.add("test")
# idphys = l1.subdomains["surfaces"]["inclusion"]
# id = gmsh.model.get_entities_for_physical_group(2,idphys)
# l1.remove(l1.dimtag(id))
# l1.build(1,finalize=False)
#
#
# l2 = copy.copy(lattice)
# idphys = l2.subdomains["surfaces"]["background"]
# id = gmsh.model.get_entities_for_physical_group(2,idphys)
# l2.remove(l2.dimtag(id))
# l2.build(1)
#

# from gyptis import dolfin as df
# import gyptis as gy
# submesh = df.SubMesh(lattice.mesh, lattice.markers, lattice.domains["background"])
# lattice.mesh=submesh
# lattice.read_mesh_info()
# del lattice.domains["inclusion"]
# # del lattice.subdomains["surfaces"]["inclusion"]
eps_inclusion = 3 - 0.1
mu_inclusion = 1
# epsilon = dict(background=1.25, inclusion=eps_inclusion)
# mu = dict(background=1, inclusion=mu_inclusion)
epsilon = dict(background=3.25)
mu = dict(background=1)
degree = 2
hom = Homogenization2D(
    lattice_bg,
    epsilon,
    mu,
    degree=degree,
)
eps_eff = hom.get_effective_permittivity()

mu_eff = hom.get_effective_permeability()

xsx


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
