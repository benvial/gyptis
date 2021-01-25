#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import pytest
from test_geometry import geom2D

from gyptis import dolfin
from gyptis.optimize import *
from gyptis.plotting import *

geom = geom2D(mesh_size=0.02)
mesh = geom.mesh_object["mesh"]
dx = geom.measure["dx"]
markers = geom.mesh_object["markers"]["triangle"]
domains = geom.subdomains["surfaces"]
r = geom.radius
l = geom.square_size


submesh = dolfin.SubMesh(mesh, markers, domains["cyl"])

W = dolfin.FunctionSpace(submesh, "DG", 0)
Wfilt = dolfin.FunctionSpace(submesh, "CG", 2)


def test_filter():
    # if __name__ == "__main__":

    f = dolfin.Expression(" sin(3*2*pi*(x[0]*x[1])/(r*r))", degree=0, r=r)
    a = project(f, W)

    values = dict(cyl=f, box=1)

    cb = dolfin.plot(a)
    plt.colorbar(cb)
    # dolfin.plot(mesh,alpha=0.8)

    plt.show()

    rfilt = r * 0.33

    af = filtering(a, rfilt)
    cb = dolfin.plot(af)
    plt.colorbar(cb)
    plt.show()

    af = filtering(a, rfilt, solver="iterative")
    cb = dolfin.plot(af)
    plt.colorbar(cb)
    plt.show()
