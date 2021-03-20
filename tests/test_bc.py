#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import numpy as np
import pytest
from test_geometry import geom2D

from gyptis import dolfin
from gyptis.bc import *

model = geom2D(mesh_size=0.01)


def test_dirichlet(model=model, tol=1e-13):
    dx = model.measure["dx"]
    ds = model.measure["ds"]
    dS = model.measure["dS"]
    W = dolfin.FunctionSpace(model.mesh_object["mesh"], "CG", 1)
    boundaries_dict = model.subdomains["curves"]
    markers_line = model.mesh_object["markers"]["line"]

    ubnd = dolfin.project(dolfin.Expression("x[0]", degree=2), W)

    bc1 = DirichletBC(W, 1, markers_line, "outer_bnds", boundaries_dict)
    bc2 = DirichletBC(W, ubnd, markers_line, "cyl_bnds", boundaries_dict)
    u = dolfin.TrialFunction(W)
    v = dolfin.TestFunction(W)
    a = dolfin.inner(dolfin.grad(u), dolfin.grad(v)) * dx + u * v * dx
    L = dolfin.Constant(0) * v * dx
    u = dolfin.Function(W)
    dolfin.solve(a == L, u, [bc1, bc2])
    a = dolfin.assemble(u * ds("outer_bnds"))
    assert np.abs(a - 4 * model.square_size) ** 2 < 1e-10

    r = model.radius
    T = np.linspace(0, 2 * np.pi, 100)
    U = []
    for t in T:
        U.append(u(r * np.cos(t), r * np.sin(t)))

    x = r * np.cos(T)
    assert np.all(np.abs(np.array(U) - x) ** 2 < 1e-6)
    assert np.mean(np.abs(np.array(U) - x) ** 2) < 1e-6
