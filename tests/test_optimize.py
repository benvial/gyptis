#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


import pytest
from test_geometry import geom2D

from gyptis import dolfin as df
from gyptis.optimize import *
from gyptis.plot import *
from gyptis.utils.helpers import array2function, rot_matrix_2d

np.random.seed(123456)

geom = geom2D(mesh_size=0.05)
mesh = geom.mesh_object["mesh"]
dx = geom.measure["dx"]
markers = geom.mesh_object["markers"]["triangle"]
domains = geom.subdomains["surfaces"]
r = geom.cyl_size


submesh = df.SubMesh(mesh, markers, domains["cyl"])

W = df.FunctionSpace(submesh, "DG", 0)
Wfilt = df.FunctionSpace(submesh, "CG", 2)
f = df.Expression(" sin(3*2*pi*(x[0]*x[1])/(r*r))", degree=0, r=r)
a0 = project(f, W)

a = array2function(np.random.rand(W.dim()), W)


def test_filter():
    af = filtering(a, 0)
    rfilt = r * 0.1
    af = filtering(a, rfilt)

    filter = Filter(rfilt)
    af = filter.apply(a)

    filter1 = Filter(rfilt, solver=filter.solver)
    filter1.apply(a)

    df.plot(af)
    rfilt_aniso = np.diag([0.458 * rfilt, 3 * rfilt])
    af_aniso = filtering(a, rfilt_aniso)

    df.plot(af_aniso)

    trot = np.pi / 3
    rot = rot_matrix_2d(trot)[:-1, :-1]
    rfilt_aniso_rot = rot.T @ rfilt_aniso @ rot
    af_aniso_rot = filtering(a, rfilt_aniso_rot)

    df.plot(af_aniso_rot)

    with pytest.raises(ValueError) as valerr:
        filtering(a, np.random.rand(5))
    assert "Wrong shape for rfilt" in str(valerr.value)


def test_simp(tol=1e-14):
    s_min, s_max, p = 4, 8, 1
    b = simp(a, s_min=s_min, s_max=s_max, p=p, complex=False)
    diff = project(b, W) - (s_min + (s_max - s_min) * a**p)
    err = assemble(abs(diff) * df.dx)
    assert err < tol
    s_min, s_max, p = 4 - 1j, 8 - 3j, 1
    np.random.seed(123456)
    A = np.random.rand(20)
    for ascalar in A:
        ascalar = 0.5
        b = simp(ascalar, s_min=s_min, s_max=s_max, p=p, complex=True)
        diff = b - (s_min + (s_max - s_min) * ascalar**p)
        err = abs(diff)
        assert err < tol


def test_projection():
    projection(a, beta=1, nu=0.5)
