#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

import numpy as np
from test_geometry import geom2D

from gyptis import dolfin
from gyptis.bc import *
from gyptis.bc import _DirichletBC
from gyptis.complex import *


def test_dirichlet():
    model = geom2D(mesh_size=0.1)
    dx = model.measure["dx"]
    ds = model.measure["ds"]
    dS = model.measure["dS"]
    W = dolfin.FunctionSpace(model.mesh_object["mesh"], "CG", 1)
    boundaries_dict = model.subdomains["curves"]
    markers_line = model.mesh_object["markers"]["line"]

    ubnd = dolfin.project(dolfin.Expression("x[0]*x[1]", degree=2), W)

    bc1 = _DirichletBC(W, 1, markers_line, "outer_bnds", boundaries_dict)
    bc2 = _DirichletBC(W, ubnd, markers_line, "cyl_bnds", boundaries_dict)
    u = dolfin.TrialFunction(W)
    v = dolfin.TestFunction(W)
    a = dolfin.inner(dolfin.grad(u), dolfin.grad(v)) * dx + u * v * dx
    L = dolfin.Constant(0) * v * dx
    u = dolfin.Function(W)
    dolfin.solve(a == L, u, [bc1, bc2])
    a = dolfin.assemble(u * ds("outer_bnds"))
    assert np.abs(a - 4 * model.square_size) ** 2 < 1e-10

    dolfin.assemble(u * dS)

    r = model.cyl_size
    T = np.linspace(-r, r, 50) / 2

    U = [u(t, -r / 2) for t in T]
    exact = -r / 2 * T
    assert np.all(np.abs(np.array(U) - exact) ** 2 < 1e-6)

    U = [u(t, r / 2) for t in T]
    exact = r / 2 * T
    assert np.all(np.abs(np.array(U) - exact) ** 2 < 1e-6)

    U = [u(-r / 2, t) for t in T]
    exact = -r / 2 * T
    assert np.all(np.abs(np.array(U) - exact) ** 2 < 1e-6)

    U = [u(r / 2, t) for t in T]
    exact = r / 2 * T
    assert np.all(np.abs(np.array(U) - exact) ** 2 < 1e-6)
