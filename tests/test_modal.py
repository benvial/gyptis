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
    import gyptis
    from gyptis import BoxPML, Scattering, c, dolfin
    from gyptis.plot import plot, plot_subdomains

    wavelength = 14
    pmesh = 3
    lmin = wavelength / pmesh

    Rx, Ry = 8, 12
    eps_cyl = 41

    geom = BoxPML(
        dim=2,
        box_size=(2 * wavelength, 2 * wavelength),
        pml_width=(2 * wavelength, 2 * wavelength),
    )
    cyl = geom.add_ellipse(0, 0, 0, Rx, Ry)
    cyl, box = geom.fragment(cyl, geom.box)
    geom.add_physical(box, "box")
    geom.add_physical(cyl, "cyl")
    [geom.set_size(pml, lmin) for pml in geom.pmls]
    geom.set_size("box", lmin)
    geom.set_size("cyl", lmin / eps_cyl**0.5)
    geom.build()

    epsilon = dict(box=1, cyl=eps_cyl)
    mu = dict(box=1, cyl=1)
    s = Scattering(
        geom,
        epsilon,
        mu,
        modal=True,
        polarization=polarization,
        degree=degree,
    )
    wavelength_target = 40
    n_eig = 6
    k_target = 2 * np.pi / wavelength_target
    solution = s.eigensolve(n_eig, k_target)
    solution["eigenvalues"]
    eig_vects = solution["eigenvectors"]
    mode = eig_vects[0]
    plot(mode.real, cmap="RdBu_r", proj_space=s.formulation.real_function_space)
    H = s.formulation.get_dual(mode, 1)
    Vvect = gyptis.dolfin.VectorFunctionSpace(geom.mesh, "CG", 2)
    H = gyptis.project(
        H,
        Vvect,
        solver_type="cg",
        preconditioner_type="jacobi",
    )
    gyptis.dolfin.plot(H.real, cmap="Greys")
    geom.plot_subdomains()
