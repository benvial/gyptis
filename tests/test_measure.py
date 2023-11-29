#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

import pytest
from numpy import pi
from test_geometry import geom2D

from gyptis import dolfin
from gyptis.measure import *

model = geom2D(mesh_size=0.1)


def test_measure(model=model, tol=1e-9):
    dx = Measure(
        "dx",
        domain=model.mesh_object["mesh"],
        subdomain_data=model.mesh_object["markers"]["triangle"],
        subdomain_dict=model.subdomains["surfaces"],
    )

    _dx = dolfin.Measure(
        "dx",
        domain=model.mesh_object["mesh"],
        subdomain_data=model.mesh_object["markers"]["triangle"],
    )
    assert dx == _dx
    area = dolfin.assemble(1 * dx)
    assert area == dolfin.assemble(1 * _dx)
    assert abs(area - model.square_size**2) < tol
    area_cyl = dolfin.assemble(1 * dx("cyl"))
    assert abs(area_cyl - model.cyl_size**2) < tol
    area_box = dolfin.assemble(1 * dx("box"))
    assert abs(area_box - (model.square_size**2 - model.cyl_size**2)) < tol

    # exterior_facets

    ds = Measure(
        "ds",
        domain=model.mesh_object["mesh"],
        subdomain_data=model.mesh_object["markers"]["line"],
        subdomain_dict=model.subdomains["curves"],
    )

    # interior_facets

    dS = Measure(
        "dS",
        domain=model.mesh_object["mesh"],
        subdomain_data=model.mesh_object["markers"]["line"],
        subdomain_dict=model.subdomains["curves"],
    )

    len_cyl = dolfin.assemble(1 * dS("cyl_bnds"))
    assert abs(len_cyl - 4 * model.cyl_size) < tol
    len_box = dolfin.assemble(1 * ds("outer_bnds"))
    assert abs(len_box - model.square_size * 4) < tol

    area = dolfin.assemble(1 * dx("everywhere"))
    assert abs(area - model.square_size**2) < tol
    area = dolfin.assemble(1 * dx(["cyl", "box"]))
    assert abs(area - model.square_size**2) < tol

    dx = Measure(
        "dx",
        domain=model.mesh_object["mesh"],
        subdomain_data=model.mesh_object["markers"]["triangle"],
        subdomain_dict=model.subdomains["surfaces"],
        subdomain_id="cyl",
    )
    area = dolfin.assemble(1 * dx)
    assert abs(area - area_cyl) < tol
