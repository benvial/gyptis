#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


from gyptis.api import BoxPML
from gyptis.complex import ComplexFunctionSpace
from gyptis.formulations import *
from gyptis.geometry import *
from gyptis.materials import *
from gyptis.sources import PlaneWave


def test_maxwell2d():
    degree = 2
    wavelength = 0.3
    pmesh = 8
    lmin = wavelength / pmesh

    geom = BoxPML(
        dim=2,
        box_size=(4 * wavelength, 4 * wavelength),
        pml_width=(wavelength, wavelength),
    )
    cyl = geom.add_circle(0, 0, 0, 0.2)
    cyl, box = geom.fragment(cyl, geom.box)
    geom.add_physical(box, "box")
    geom.add_physical(cyl, "cyl")
    [geom.set_size(pml, lmin) for pml in geom.pmls]
    geom.set_size("box", lmin)
    geom.set_size("cyl", lmin)
    geom.build()
    mesh = geom.mesh_object["mesh"]

    epsilon = dict(box=1, cyl=3)
    mu = dict(box=1, cyl=1)

    stretch = 1 - 1j
    pmlx = PML("x", stretch=stretch, matched_domain="box", applied_domain="pmlx")
    pmly = PML("y", stretch=stretch, matched_domain="box", applied_domain="pmly")
    pmlxy = PML("xy", stretch=stretch, matched_domain="box", applied_domain="pmlxy")

    epsilon_coeff = Coefficient(epsilon, geometry=geom, pmls=[pmlx, pmly, pmlxy])
    mu_coeff = Coefficient(mu, geometry=geom, pmls=[pmlx, pmly, pmlxy])

    coefficients = epsilon_coeff, mu_coeff
    V = ComplexFunctionSpace(mesh, "CG", degree)

    pw = PlaneWave(wavelength=wavelength, angle=0, dim=2, domain=mesh, degree=degree)

    bcs = {}

    Maxwell2D(
        geom,
        coefficients,
        V,
        source=pw,
        boundary_conditions=bcs,
        source_domains="cyl",
        reference="box",
    )

    ###########################

    geom = BoxPML(
        dim=2,
        box_size=(4 * wavelength, 4 * wavelength),
        pml_width=(wavelength, wavelength),
    )
    cyl = geom.add_circle(0, 0, 0, 0.2)
    box = geom.cut(geom.box, cyl)
    geom.add_physical(box, "box")
    cyl_bnds = geom.get_boundaries("box")[-1]
    geom.add_physical(cyl_bnds, "cyl_bnds", dim=1)
    [geom.set_size(pml, lmin) for pml in geom.pmls]
    geom.set_size("box", lmin)
    geom.build(0)
    mesh = geom.mesh_object["mesh"]
    epsilon = dict(box=1, cyl=3)
    mu = dict(box=1, cyl=1)

    stretch = 1 - 1j
    pmlx = PML("x", stretch=stretch, matched_domain="box", applied_domain="pmlx")
    pmly = PML("y", stretch=stretch, matched_domain="box", applied_domain="pmly")
    pmlxy = PML("xy", stretch=stretch, matched_domain="box", applied_domain="pmlxy")

    epsilon_coeff = Coefficient(epsilon, geometry=geom, pmls=[pmlx, pmly, pmlxy])
    mu_coeff = Coefficient(mu, geometry=geom, pmls=[pmlx, pmly, pmlxy])

    coefficients = epsilon_coeff, mu_coeff
    function_space = ComplexFunctionSpace(mesh, "CG", degree)

    pw = PlaneWave(wavelength=wavelength, angle=0, dim=2, domain=mesh, degree=degree)

    bcs = {"cyl_bnds": "PEC"}

    maxwell = Maxwell2D(
        geom,
        coefficients,
        function_space,
        source=pw,
        boundary_conditions=bcs,
        source_domains="cyl",
        reference="box",
    )
    maxwell.build_boundary_conditions()


def test_maxwell2d_periodic():
    lambda0, period = 1, 1

    from gyptis.models.grating2d import Layered2D, OrderedDict
    from gyptis.sources.stack import make_stack

    thicknesses = OrderedDict(
        {
            "pml_bottom": lambda0,
            "substrate": lambda0,
            "groove": lambda0,
            "superstrate": lambda0,
            "pml_top": lambda0,
        }
    )

    degree = 1
    geom = Layered2D(period, thicknesses)
    geom.build()
    epsilon = dict(
        {
            "substrate": 3,
            "groove": 1,
            "superstrate": 1,
        }
    )
    mu = dict(
        {
            "substrate": 1,
            "groove": 1,
            "superstrate": 1,
        }
    )
    stretch = 1 - 1j
    pml_top = PML(
        "y", stretch=stretch, matched_domain="superstrate", applied_domain="pml_top"
    )
    pml_bottom = PML(
        "y", stretch=stretch, matched_domain="substrate", applied_domain="pml_bottom"
    )
    epsilon_coeff = Coefficient(epsilon, geometry=geom, pmls=[pml_top, pml_bottom])
    mu_coeff = Coefficient(mu, geometry=geom, pmls=[pml_top, pml_bottom])
    coefficients = epsilon_coeff, mu_coeff
    function_space = ComplexFunctionSpace(geom.mesh, "CG", degree)
    pw = PlaneWave(
        wavelength=lambda0, angle=np.pi / 4, dim=2, domain=geom.mesh, degree=degree
    )

    make_stack(
        geom,
        coefficients,
        pw,
        polarization="TM",
        source_domains=["groove"],
        degree=1,
        dim=2,
    )
    bcs = {}
    maxwell_per2d = Maxwell2DPeriodic(
        geom,
        coefficients,
        function_space,
        source=pw,
        boundary_conditions=bcs,
        source_domains=["groove"],
        reference="superstrate",
    )

    maxwell_per2d.build_lhs()
    maxwell_per2d.build_rhs()
