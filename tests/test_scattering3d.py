#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


"""
Introduction to electromagnetic scattering: tutorial
https://www.osapublishing.org/josaa/fulltext.cfm?uri=josaa-35-1-163&id=380136
"""


import numpy as np
import pytest


def test_scatterring3d(shared_datadir):
    from gyptis import c, dolfin, epsilon_0, mu_0
    from gyptis.complex import Complex, assemble, cross, curl, dot
    from gyptis.models.scattering3d import BoxPML3D, Scatt3D
    from gyptis.sources import PlaneWave

    dolfin.parameters["form_compiler"]["quadrature_degree"] = 2

    degree = 1
    pmesh = 1
    pmesh_scatt = 1 * pmesh
    eps_sphere = 4
    SCSN = []
    Gamma = np.linspace(0.1, 5, 100)
    Gamma = [2]
    R_sphere = 0.25
    for gamma in Gamma:
        circ = 2 * np.pi * R_sphere
        lambda0 = circ / gamma
        b = R_sphere * 3
        box_size = (b, b, b)
        pml_width = (lambda0, lambda0, lambda0)
        g = BoxPML3D(box_size=box_size, pml_width=pml_width)
        radius_cs_sphere = 0.8 * min(g.box_size) / 2
        box = g.box
        sphere = g.add_sphere(0, 0, 0, R_sphere)
        sphere_cross_sections = g.add_sphere(0, 0, 0, radius_cs_sphere)
        sphere, sphere_cross_sections, box = g.fragment(
            sphere, [sphere_cross_sections, box]
        )

        g.add_physical([box, sphere_cross_sections], "box")
        g.add_physical(sphere, "sphere")
        surf = g.get_boundaries(sphere_cross_sections, physical=False)[0]
        g.add_physical(surf, "calc", dim=2)
        smin = 1 * R_sphere / 2
        s = min(lambda0 / pmesh, smin)
        smin_pml = lambda0 / (0.66 * pmesh)
        for coord in ["x", "y", "z", "xy", "xz", "yz", "xyz"]:
            g.set_mesh_size({f"pml{coord}": smin_pml})

        g.set_size(box, s)
        g.set_size(sphere_cross_sections, s)
        g.set_size(surf, s, dim=2)
        s = min(lambda0 / (eps_sphere**0.5 * pmesh_scatt), smin)
        g.set_size(sphere, s)
        g.build()

        epsilon = dict(sphere=eps_sphere, box=1)
        mu = dict(sphere=1, box=1)

        pw = PlaneWave(
            wavelength=lambda0, angle=(0, 0, 0), dim=3, domain=g.mesh, degree=degree
        )
        bcs = {}
        s = Scatt3D(
            g,
            epsilon,
            mu,
            pw,
            boundary_conditions=bcs,
            degree=degree,
        )

        s.solve()
        Z0 = np.sqrt(mu_0 / epsilon_0)
        S0 = 1 / (2 * Z0)
        n_out = g.unit_normal_vector
        Es = s.solution["diffracted"]
        inv_mu_coeff = s.coefficients[1].invert().as_subdomain()
        omega = s.source.pulsation
        Hs = inv_mu_coeff / Complex(0, dolfin.Constant(omega * mu_0)) * curl(Es)
        Ps = dolfin.Constant(0.5) * cross(Es, Hs.conj).real
        Ws = -assemble(dot(n_out, Ps)("+") * s.dS("calc"))
        Sigma_s = Ws / S0
        S_sphere = R_sphere**2 * np.pi
        Sigma_s_norm = Sigma_s / S_sphere
        SCSN.append(Sigma_s_norm)
