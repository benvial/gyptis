#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

import os

import numpy as np
import pytest

# dolfin.set_log_level(0)

ref = [1.0022840, 1.0374103, 1.0950892, 1.1977349]
r = [0.1, 0.25, 0.34, 0.43]


@pytest.mark.parametrize("i", range(4))
def test(i):
    from gyptis import dolfin
    from gyptis.models import Homogenization3D, Lattice3D

    dolfin.parameters["form_compiler"]["quadrature_degree"] = 5
    a = 1
    v = (a, 0, 0), (0, a, 0), (0, 0, a)
    R = r[i] * a
    lmin = 0.1
    eps_incl = 3

    lattice = Lattice3D(v, verbose=0)
    incl = lattice.add_sphere(a / 2, a / 2, a / 2, R)
    incl, cell = lattice.fragment(incl, lattice.cell)
    lattice.add_physical(cell, "background")
    lattice.add_physical(incl, "inclusion")
    bnd = lattice.get_boundaries(incl)
    lattice.add_physical(bnd, "bnd", 2)
    lattice.set_size("background", lmin)
    lattice.set_size("inclusion", lmin)
    lattice.set_size("bnd", lmin / 1, 2)
    # lattice.remove_all_duplicates()
    lattice.build()

    degree = 2
    epsilon = dict(background=1, inclusion=eps_incl)
    mu = dict(background=1, inclusion=1)

    hom = Homogenization3D(
        lattice,
        epsilon,
        mu,
        degree=degree,
        direct=False,
    )
    #
    eps_eff = hom.get_effective_permittivity()
    # print(eps_eff)
    n_eff = (eps_eff[0][0].real) ** 0.5
    print(ref[i])
    print(n_eff)

    assert np.allclose(n_eff, ref[i], rtol=6e-3)

    # eps = hom.formulation.epsilon.as_subdomain()
    # eps_mean = hom.unit_cell_mean(eps)
    #
    # phix = hom.solution["epsilon"]["x"]
    # n = hom.geometry.unit_normal_vector
    # psix1 = (eps_incl-1) * gy.assemble((n[0] * phix)("+")*hom.dS("bnd"))
    # eps_eff1 = eps_mean - psix1
    # print(eps_eff1.real ** 0.5)

    epsilon = dict(background=1, inclusion=1)
    mu = dict(background=1, inclusion=eps_incl)
    hom = Homogenization3D(
        lattice,
        epsilon,
        mu,
        degree=degree,
        direct=False,
    )
    #
    mu_eff = hom.get_effective_permeability()
    n_eff = (mu_eff[0][0].real) ** 0.5
    print(n_eff)
    assert np.allclose(n_eff, ref[i], rtol=6e-3)

    print("-----")
