#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import dolfin as df
import pytest
from numpy import pi
from test_geometry import geom2D

from gyptis.helpers import *

# m.geometric_dimension()


model = geom2D(mesh_size=0.01)


# tol=1e-13

# if __name__=="__main__":
def test_measure(model=model, tol=1e-13):
    dx = Measure(
        "dx",
        domain=model.mesh_object["mesh"],
        subdomain_data=model.mesh_object["markers"]["triangle"],
        subdomain_dict=model.subdomains["surfaces"],
    )

    _dx = df.Measure(
        "dx",
        domain=model.mesh_object["mesh"],
        subdomain_data=model.mesh_object["markers"]["triangle"],
    )
    assert dx == _dx
    area = df.assemble(1 * dx)
    assert area == df.assemble(1 * _dx)
    assert abs(area - model.square_size ** 2) < tol
    area_cyl = df.assemble(1 * dx("cyl"))
    assert abs(area_cyl - pi * model.radius ** 2) < 1e-4
    area_box = df.assemble(1 * dx("box"))
    assert abs(area_box - (model.square_size ** 2 - pi * model.radius ** 2)) < 1e-4

    ## exterior_facets

    ds = Measure(
        "ds",
        domain=model.mesh_object["mesh"],
        subdomain_data=model.mesh_object["markers"]["line"],
        subdomain_dict=model.subdomains["curves"],
    )

    ## interior_facets

    dS = Measure(
        "dS",
        domain=model.mesh_object["mesh"],
        subdomain_data=model.mesh_object["markers"]["line"],
        subdomain_dict=model.subdomains["curves"],
    )

    len_circ = df.assemble(1 * dS("cyl_bnds"))
    assert abs(len_circ - 2 * pi * model.radius) < 1e-4
    len_box = df.assemble(1 * ds("outer_bnds"))
    assert abs(len_box - model.square_size * 4) < 1e-4


def test_dirichlet(model=model, tol=1e-13):
    dx = model.measure["dx"]
    ds = model.measure["ds"]
    dS = model.measure["dS"]
    W = df.FunctionSpace(model.mesh_object["mesh"], "CG", 1)
    boundaries_dict = model.subdomains["curves"]
    markers_line = model.mesh_object["markers"]["line"]

    ubnd = df.project(df.Expression("x[0]", degree=2), W)

    bc1 = DirichletBC(W, 1, markers_line, "outer_bnds", boundaries_dict)
    bc2 = DirichletBC(W, ubnd, markers_line, "cyl_bnds", boundaries_dict)
    u = df.TrialFunction(W)
    v = df.TestFunction(W)
    a = df.inner(df.grad(u), df.grad(v)) * dx + u * v * dx
    L = df.Constant(0) * v * dx
    u = df.Function(W)
    df.solve(a == L, u, [bc1, bc2])
    a = df.assemble(u * ds("outer_bnds"))
    assert np.abs(a - 4 * model.square_size) ** 2 < 1e-10

    r = model.radius
    T = np.linspace(0, 2 * np.pi, 100)
    U = []
    for t in T:
        U.append(u(r * np.cos(t), r * np.sin(t)))

    x = r * np.cos(T)
    assert np.all(np.abs(np.array(U) - x) ** 2 < 1e-6)
    assert np.mean(np.abs(np.array(U) - x) ** 2) < 1e-6
