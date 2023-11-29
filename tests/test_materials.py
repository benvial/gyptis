#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


import pytest
from numpy import e, pi
from test_geometry import geom2D

import gyptis
from gyptis.api import BoxPML
from gyptis.complex import *
from gyptis.geometry import *
from gyptis.materials import *

model = geom2D(mesh_size=0.1)
mesh = model.mesh_object["mesh"]
dx = model.measure["dx"]
r = model.cyl_size
lsq = model.square_size
markers = model.mesh_object["markers"]["triangle"]
domains = model.subdomains["surfaces"]


W = dolfin.FunctionSpace(mesh, "CG", 1)


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_subdomain(degree):
    tol = 1e-2
    values = dict(cyl=12, box=1)
    sub = Subdomain(markers, domains, values, degree=degree)
    sub_py = Subdomain(markers, domains, values, degree=degree, cpp=False)

    a = dolfin.assemble(sub * dx)
    a_py = dolfin.assemble(sub_py * dx)
    assert a == a_py

    a_cyl = r**2
    a_box = lsq**2 - a_cyl
    a_test = a_cyl * values["cyl"] + a_box * values["box"]

    assert abs(a - a_test) ** 2 < tol

    f = dolfin.Expression("(x[0]*x[0]+x[1]*x[1])", degree=degree, r=r)

    values = dict(cyl=f, box=1)
    sub_with_function = Subdomain(markers, domains, values, degree=degree)
    integral = dolfin.assemble(sub_with_function * dx("cyl"))
    Iexact = 4 / 3 * r**3
    assert abs(integral - Iexact) ** 2 < tol
    sub_with_function_python = Subdomain(
        markers, domains, values, degree=degree, cpp=False
    )
    I_python = dolfin.assemble(sub_with_function_python * dx("cyl"))
    assert abs(I_python - Iexact) ** 2 < tol


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_subdomain_complex(degree):
    tol = 1e-15
    values = dict(cyl=12 + 2j, box=1)
    sub = Subdomain(markers, domains, values, degree=degree)
    sub_py = Subdomain(
        markers,
        domains,
        values,
        degree=degree,
        cpp=False,
    )

    a = assemble(sub * dx)
    a_py = assemble(sub_py * dx)
    assert a == a_py

    a_cyl = r**2
    a_box = lsq**2 - a_cyl
    a_test = a_cyl * values["cyl"] + a_box * values["box"]
    tol1 = 1e-2
    assert abs(a - a_test) ** 2 < tol1
    f = dolfin.Expression(" exp(-pow(x[0]/r,2) - pow(x[1]/r,2))", degree=degree, r=r)

    eps_cyl = 2
    eps_box = f
    mapping = dict(cyl=eps_cyl, box=eps_box)
    eps = Subdomain(markers, domains, mapping, degree=degree)
    assert assemble((eps - eps_box) * dx("box")) < tol
    assert assemble((eps - eps_cyl) * dx("cyl")) < tol

    eps_cyl = 2 - 1.2 * 1j
    eps_box = Complex(f, f)
    mapping = dict(cyl=eps_cyl, box=eps_box)
    eps = Subdomain(markers, domains, mapping, degree=degree)
    assert assemble(abs(eps - eps_box) * dx("box")) < tol
    assert assemble(abs(eps - eps_cyl) * dx("cyl")) < tol

    eps_cyl = [[1, 2], [3, 4]]
    eps_box = 2
    mapping = dict(cyl=eps_cyl, box=eps_box)
    eps = Subdomain(markers, domains, mapping, degree=degree)
    eps_box_tens = eps_box * np.eye(2)
    for i in [0, 1]:
        for j in [0, 1]:
            assert assemble(abs(eps[i][j] - eps_box_tens[i, j]) * dx("box")) < tol
            assert assemble(abs(eps[i][j] - eps_cyl[i][j]) * dx("cyl")) < tol

    eps_cyl = [[1, 2 - 1j], [3, Complex(f, f)]]
    eps_box = 1
    mapping = dict(cyl=eps_cyl, box=eps_box)
    eps = Subdomain(markers, domains, mapping, degree=degree)
    eps_box_tens = eps_box * np.eye(2)
    for i in [0, 1]:
        for j in [0, 1]:
            assert assemble(abs(eps[i][j] - eps_box_tens[i, j]) * dx("box")) < tol
            assert assemble(abs(eps[i][j] - eps_cyl[i][j]) * dx("cyl")) < tol


def test_pml():
    PML()


def test_coefficient():
    geom = BoxPML(dim=2)
    cyl = geom.add_circle(0, 0, 0, 0.2)
    cyl, box = geom.fragment(cyl, geom.box)
    geom.add_physical(box, "box")
    geom.add_physical(cyl, "cyl")
    geom.set_size("cyl", 0.04)
    geom.build()

    epsilon = dict(box=1, cyl=3)
    eps = Coefficient(epsilon)

    pmlx = PML("x", stretch=1 - 1j, matched_domain="box", applied_domain="pmlx")
    pmly = PML("y", stretch=1 - 1j, matched_domain="box", applied_domain="pmly")
    pmlxy = PML("xy", stretch=1 - 1j, matched_domain="box", applied_domain="pmlxy")

    eps = Coefficient(epsilon, geometry=geom, pmls=[pmlx, pmly, pmlxy])
    eps.apply_pmls()

    eps.as_subdomain()

    # eps.plot()
    eps.plot(component=(1, 1))
    annex = eps.build_annex(domains=["cyl"], reference="box")

    # eps.plot()
    annex.plot(component=(1, 1))

    eps = Coefficient(epsilon, geometry=geom, pmls=[pmlx, pmly, pmlxy])

    eps.as_property()
    eps.as_property(dim=3)

    eps.to_xi()
    eps.to_chi()


# TODO: be careful here
# Warning
# /ufl/exproperators.py:336: FutureWarning: elementwise comparison failed;
# returning scalar instead, but in the future will perform elementwise comparison
# if arg in ("+", "-"):
# possible solution: check f.ufl_operands and project first
