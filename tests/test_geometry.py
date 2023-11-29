#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

from math import pi

import pytest

from gyptis import BoxPML, dolfin
from gyptis.geometry import *


def geom2D(square_size=1, cyl_size=0.3, mesh_size=0.1):
    model = Geometry("Square", dim=2)
    box = model.add_rectangle(
        -square_size / 2, -square_size / 2, 0, square_size, square_size
    )
    cyl = model.add_rectangle(-cyl_size / 2, -cyl_size / 2, 0, cyl_size, cyl_size)
    cyl, box = model.fragment(cyl, box)

    model.add_physical(box, "box")
    model.add_physical(cyl, "cyl")

    outer_bnds = model.get_boundaries("box")[:-4]
    cyl_bnds = model.get_boundaries("cyl")
    model.add_physical(outer_bnds, "outer_bnds", dim=1)
    model.add_physical(cyl_bnds, "cyl_bnds", dim=1)
    model.set_size("box", 1 * mesh_size)
    model.set_size("cyl", 1 * mesh_size)
    model.set_size("cyl_bnds", mesh_size, dim=1)
    model.build()
    model.cyl_size = cyl_size
    model.square_size = square_size

    return model


def test_2D():
    model = geom2D(mesh_size=0.01)
    dx = model.measure["dx"]
    area_cyl = dolfin.assemble(1 * dx("cyl"))
    assert abs(area_cyl - model.cyl_size**2) < 1e-4


def test_3D():
    model = Geometry("Sphere")
    box = model.add_box(-1, -1, -1, 2, 2, 2)
    sphere = model.add_sphere(0, 0, 0, 0.5)
    sphere, box = model.fragment(sphere, box)
    model.set_size(box, 0.3)
    model.set_size(sphere, 0.3)
    model.add_physical(sphere, "sphere")
    model.add_physical(box, "box")
    outer_bnds = model.get_boundaries("box")
    model.add_physical(outer_bnds, "outer_bnds", dim=2)
    model.build()
    assert model.subdomains == {
        "volumes": {"sphere": 1, "box": 2},
        "surfaces": {"outer_bnds": 3},
        "curves": {},
        "points": {},
    }


def test_box_pml():
    with pytest.raises(ValueError):
        BoxPML(dim=1)


def test_box_pml_2D():
    model = BoxPML(dim=2)
    cyl = model.add_disk(0, 0, 0, 0.1, 0.1)
    cyl, model.box = model.fragment(cyl, model.box)
    model.add_physical(cyl, "cyl")
    model.add_physical(model.box, "box")
    model.set_size(model.box, 0.03)
    model.set_size(model.pmls, 0.1)
    model.set_size(cyl, 0.01)
    model.build()


def test_box_pml_3D():
    model = BoxPML(dim=3)
    sphere = model.add_sphere(0, 0, 0, 0.1)
    cyl, model.box = model.fragment(sphere, model.box)
    model.add_physical(sphere, "sphere")
    model.add_physical(model.box, "box")
    model.set_size(model.box, 0.06)
    model.set_size(model.pmls, 0.1)
    model.set_size(sphere, 0.04)
    model.build()


def test_spline():
    g = Geometry(dim=2)
    N = 20
    t = np.linspace(0, 2 * np.pi, N)
    r = np.random.rand(N) + 1
    points = np.array([r * np.cos(t), r * np.sin(t), np.zeros_like(t)]).T
    points[0, :] = points[-1, :]
    size = 0.2
    spl = g.add_spline(points, size)
    g.add_physical(spl, "kk")

    g.build()
