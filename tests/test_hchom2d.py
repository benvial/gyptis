#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


import numpy as np

import gyptis as gy


def test_hchom2d():
    d = 1
    v = (d, 0), (0, d)
    a = d / 2
    lmin = a / 50

    lattice = gy.Lattice(dim=2, vectors=v)
    incl = lattice.add_square(a / 2, a / 2, 0, a)
    cell = lattice.cut(lattice.cell, incl)
    lattice.add_physical(cell, "background")
    lattice.set_size("background", lmin)
    lattice.build()

    inclusion = gy.Geometry(dim=2)
    incl = inclusion.add_square(a / 2, a / 2, 0, a)
    inclusion.add_physical(incl, "inclusion")
    bnds = inclusion.get_boundaries("inclusion")
    inclusion.add_physical(bnds, "inclusion_bnds", dim=1)
    inclusion.set_size("inclusion", lmin)
    inclusion.build()

    eps_i = 200 - 5j
    epsilon = dict(inclusion=eps_i, background=1)
    mu = dict(inclusion=1, background=1)

    def analytical_mueff(k, Nmax=10):
        nms = [2 * n + 1 for n in range(Nmax)]
        mu = 1
        for n in nms:
            for m in nms:
                knm = np.pi / a * (n**2 + m**2) ** 0.5
                qn = np.pi / a * n
                pm = np.pi / a * m
                alpha = 2 / qn * 2 / pm
                norm = (a / 2) ** 2
                mu += (
                    -(k**2 * eps_i) / (k**2 * eps_i - knm**2) * alpha**2 / norm
                )
        return mu

    hom = gy.models.HighContrastHomogenization2D(
        lattice, inclusion, epsilon, mu, degree=2
    )
    eps_eff = hom.get_effective_permittivity(scalar=True)
    print(eps_eff)
    assert np.abs(eps_eff[0][0] - 1.73) < 1e-3
    neigs = 50
    wavevector_target = 0.1
    lambdas = np.linspace(4, 15, 1000) * d
    k = 2 * np.pi / lambdas

    mu_eff = hom.get_effective_permeability(
        k, neigs=neigs, wavevector_target=wavevector_target
    ).tocomplex()
    mu_eff_ana = analytical_mueff(k)
    err = np.mean(np.abs(mu_eff - mu_eff_ana))
    print(err)
    assert np.allclose(mu_eff, mu_eff_ana, atol=5e-3)
    assert err < 1e-3
