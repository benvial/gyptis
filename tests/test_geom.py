#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import pytest
from gyptis.geometry import *
import dolfin as df
from numpy import pi

def geom2D(square_size=1, radius=0.3, mesh_size=0.1):
    model = Model("Cylinder", dim=2)
    box = model.addRectangle(
        -square_size / 2, -square_size / 2, 0, square_size, square_size
    )
    outer_bnds = model.get_boundaries(box)
    cyl = model.addDisk(0, 0, 0, radius, radius)
    cyl, box = model.fragmentize(cyl, box)
    cyl_bnds = model.get_boundaries(cyl)
    model.set_size(box, mesh_size)
    model.set_size(cyl, mesh_size)
    model.add_physical(cyl, "cyl")
    model.add_physical(box, "box")
    model.add_physical(outer_bnds, "outer_bnds", dim=1)
    model.add_physical(cyl_bnds, "cyl_bnds", dim=1)
    mesh_object = model.build()
    model.radius = radius
    model.square_size = square_size

    return model


def test_2D():
    model = geom2D(mesh_size=0.01)
    assert model.subdomains == {
        "volumes": {},
        "surfaces": {"cyl": 1, "box": 2},
        "curves": {"outer_bnds": 3, "cyl_bnds": 4},
        "points": {},
    }
    dx = model.measure["dx"]
    area_cyl = df.assemble(1 * dx("cyl"))
    assert abs(area_cyl - pi * model.radius ** 2) < 1e-4


def test_3D():
    model = Model("Sphere")
    box = model.addBox(-1, -1, -1, 2, 2, 2)
    sphere = model.addSphere(0, 0, 0, 0.5)
    sphere, box = model.fragmentize(sphere, box)
    model.set_size(box, 0.3)
    model.set_size(sphere, 0.3)
    outer_bnds = model.get_boundaries(box)
    model.add_physical(sphere, "sphere")
    model.add_physical(box, "box")
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
    cyl = model.addDisk(0, 0, 0, 0.1, 0.1)
    cyl, model.box = model.fragmentize(cyl, model.box)
    model.add_physical(cyl, "cyl")
    model.add_physical(model.box, "box")
    model.set_size(model.box, 0.03)
    model.set_size(model.pmls, 0.1)
    model.set_size(cyl, 0.01)
    mesh_object = model.build()

    


def test_box_pml_3D():
    model = BoxPML(dim=3)
    sphere = model.addSphere(0, 0, 0, 0.1)
    cyl, model.box = model.fragmentize(sphere, model.box)
    model.add_physical(sphere, "sphere")
    model.add_physical(model.box, "box")
    model.set_size(model.box, 0.06)
    model.set_size(model.pmls, 0.1)
    model.set_size(sphere, 0.04)
    mesh_object = model.build()

# 
