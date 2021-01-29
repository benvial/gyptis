#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import pytest
from test_geometry import geom2D

from gyptis import dolfin as df
from gyptis.helpers import array2function, rot_matrix_2d
from gyptis.optimize import *
from gyptis.plotting import *

np.random.seed(123456)

geom = geom2D(mesh_size=0.05)
mesh = geom.mesh_object["mesh"]
dx = geom.measure["dx"]
markers = geom.mesh_object["markers"]["triangle"]
domains = geom.subdomains["surfaces"]
r = geom.radius
l = geom.square_size


submesh = df.SubMesh(mesh, markers, domains["cyl"])

W = df.FunctionSpace(submesh, "DG", 0)
Wfilt = df.FunctionSpace(submesh, "CG", 2)
f = df.Expression(" sin(3*2*pi*(x[0]*x[1])/(r*r))", degree=0, r=r)
a0 = project(f, W)

a = array2function(np.random.rand(W.dim()), W)


def test_filter():
    # if __name__ == "__main__":
    values = dict(cyl=f, box=1)
    af = filtering(a, 0)
    rfilt = r * 0.1
    af = filtering(a, rfilt)

    af = filtering(a, rfilt, solver="direct")
    plt.figure()
    df.plot(af)
    plt.title("isotropic")
    rfilt_aniso = np.diag([0.458 * rfilt, 3 * rfilt])
    af_aniso = filtering(a, rfilt_aniso, solver="iterative")

    plt.figure()
    df.plot(af_aniso)
    plt.title("anisotropic")

    trot = np.pi / 3
    rot = rot_matrix_2d(trot)[:-1, :-1]
    rfilt_aniso_rot = rot.T @ rfilt_aniso @ rot
    af_aniso_rot = filtering(a, rfilt_aniso_rot, solver="iterative")
    plt.figure()
    df.plot(af_aniso_rot)
    plt.title("anisotropic rotated")
    with pytest.raises(ValueError) as valerr:
        filtering(a, np.random.rand(5))
    assert "Wrong shape for rfilt" in str(valerr.value)


def test_simp(tol=1e-14):
    s_min, s_max, p = 4, 8, 1
    b = simp(a, s_min=s_min, s_max=s_max, p=p)
    diff = project(b, W) - (s_min + (s_max - s_min) * a ** p)
    err = assemble(abs(diff) * df.dx)
    assert err < tol


def test_projection():
    projection(a, beta=1, nu=0.5)
