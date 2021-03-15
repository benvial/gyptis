#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from math import pi

import pytest

from gyptis import dolfin
from gyptis.geometry import *


def geom2D(square_size=1, radius=0.3, mesh_size=0.1):
    model = Geometry("Cylinder", dim=2)
    box = model.add_rectangle(
        -square_size / 2, -square_size / 2, 0, square_size, square_size
    )
    cyl = model.add_disk(0, 0, 0, radius, radius)
    cyl, box = model.fragment(cyl, box)
    model.synchronize()

    model.add_physical(box, "box")
    model.add_physical(cyl, "cyl")

    outer_bnds = model.get_boundaries("box")[:-1]
    cyl_bnds = model.get_boundaries("cyl")
    model.add_physical(outer_bnds, "outer_bnds", dim=1)
    model.add_physical(cyl_bnds, "cyl_bnds", dim=1)
    model.set_size("box", 1 * mesh_size)
    model.set_size("cyl", 1 * mesh_size)
    model.set_size("cyl_bnds", mesh_size, dim=1)
    mesh_object = model.build(interactive=False)
    model.radius = radius
    model.square_size = square_size

    return model


def test_2D():
    model = geom2D(mesh_size=0.01)
    # assert model.subdomains == {
    #     "volumes": {},
    #     "surfaces": {"cyl": 1, "box": 2},
    #     "curves": {"outer_bnds": 3, "cyl_bnds": 4},
    #     "points": {},
    # }
    dx = model.measure["dx"]
    area_cyl = dolfin.assemble(1 * dx("cyl"))
    print(area_cyl)
    print(pi * model.radius ** 2)
    assert abs(area_cyl - pi * model.radius ** 2) < 1e-4


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
    mesh_object = model.build()
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
    mesh_object = model.build()


def test_box_pml_3D():
    model = BoxPML(dim=3)
    sphere = model.add_sphere(0, 0, 0, 0.1)
    cyl, model.box = model.fragment(sphere, model.box)
    model.add_physical(sphere, "sphere")
    model.add_physical(model.box, "box")
    model.set_size(model.box, 0.06)
    model.set_size(model.pmls, 0.1)
    model.set_size(sphere, 0.04)
    mesh_object = model.build()


def test_api():
    g = Geometry(dim=2)

    c1 = g.Rectangle((0, 0, 0), (2, 2.5), name="c2")

    c2 = g.Circle((1.6, 2, 0), 0.8, name="c1")
    c3 = c1 - c2
    c4 = g.Circle((1.6, 2, 0), 0.65)
    c5 = g.Rectangle((-2, 0.8, 0), (2.8, 0.3))
    c5.rotate((0.5, 0.5, 0), (0, 0, 1), -0.8)
    c3 -= c5
    r = g.Rectangle((-0.9, -0.8, 0), (4, 4))
    c3, c4, r = (c3 + c4) / r
    r << "background"
    r | 0.5
    c3 << "body"
    c3 | 0.2
    c4 << "circle"
    c4 | 0.07
    # c3.extrude((0,0,1),num_el=[10])
    # c4.extrude((0,0,2),num_el=[20])
    # r.extrude((0,0,0.2),num_el=[3])
    # #
    # # out = occ.extrude(g.dimtag([1,2,3]),0,0,1,numElements=[10])
    # # occ.synchronize()
    #
    # g.dim = 3

    g.build()
    return g

# 
# import importlib
# from gyptis import geometry
# 
# importlib.reload(geometry)
# 
# from gyptis.geometry import *

def test_ellipse():
    g = Geometry(dim=2)

    c1 = g.add_ellipse(0,0,0,0.2,0.4)
    # g.add_physical(c1,"kk")

    c2 = g.add_ellipse(0,0,0,0.6,0.4)
    c3 = g.add_ellipse(0,0,0,0.4,0.4)

    g.build(0,1,0,0,0)

    g = Geometry(dim=2)

    c1 = g.Ellipse((0, 0, 0), (0.2, 0.3))
    # g.add_physical(c1,"kk")
    c2 = g.Ellipse((0, 0, 0), (0.6, 0.4))
    c2 -= c1

    c3 = g.Ellipse((-0.4, 0, 0), (0.1, 0.1))
    c4 = g.Ellipse((0.4, 0, 0), (0.1, 0.1))
    a, b,c = c2 / (c3+c4)

    a << "a"
    b << "b"
    c << "c"
    b | 0.02
    a | 0.01

    g.build()


    g = Geometry(dim=2)

    c1 = g.Circle((0, 0, 0), 1)
    # g.add_physical(c1,"kk")
    c2=[]
    for t in np.linspace(0,2*np.pi,13)[:-1]:
        c1 -= g.Ellipse((np.cos(t), np.sin(t), 0), (0.2, 0.2))
        c1 /= g.Ellipse((0.5*np.cos(t), 0.5*np.sin(t), 0), (0.07, 0.07))

        c2.append( c1[0])
        print(len(c1))
        c1 = c1[1]

    c2 = sum(c2[1:],start=c2[0])

    print(c2.get_boundaries())

    c2 << "c2"
    c1 << "c1"


    g.build()

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
