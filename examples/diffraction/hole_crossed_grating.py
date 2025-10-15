#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.1
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
3D Checkerboard Grating
=======================

Example of a dielectric bi-periodic diffraction grating.
"""
# sphinx_gallery_thumbnail_number = 2


import pprint
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy

gy.dolfin.parameters["form_compiler"]["quadrature_degree"] = 5
# gy.dolfin.parameters["ghost_mode"] = "shared_facet"
gy.dolfin.set_log_level(100)

##############################################################################
# Structure is the same as in :cite:p:`Demesy2010`.
#
# The units of lengths are in nanometers here, and we first define some
# geometrical and optical parameters:


lambda0 = 0.5
dx = dy = 1  # 5 * lambda0 * 2 ** 0.5 / 4  # periods of the grating
h = 0.05
theta0 = 0
phi0 = 0
psi0 = gy.pi / 4
eps_diel = 2.25
eps_layer = 0.8125 - 5.2500j

##############################################################################
# The thicknesses of the different layers are specified with an
# ``OrderedDict`` object **from bottom to top**:

thicknesses = OrderedDict(
    {
        "pml_bottom": lambda0,
        "substrate": lambda0 / 1,
        "groove": 1 * h,
        "superstrate": lambda0 / 1,
        "pml_top": lambda0,
    }
)

##############################################################################
# Here we set the mesh refinement parameters, in order to be able to have
# ``parmesh`` cells per wavelength of the field inside each subdomain

degree = 2
pmesh = 3
pmesh_hole = pmesh * 1
mesh_param = dict(
    {
        "pml_bottom": 1 * pmesh * eps_diel.real**0.5,
        "substrate": pmesh * eps_diel.real**0.5,
        "groove": pmesh * abs(eps_layer) ** 0.5,
        "hole": pmesh_hole,
        "superstrate": pmesh,
        "pml_top": 1 * pmesh,
    }
)

##############################################################################
# Let's create the geometry using the :class:`~gyptis.Layered`
# class:
geom = gy.Layered(3, (dx, dy), thicknesses)
z0 = geom.z_position["groove"]  # + h/10
# l_pillar = 0.9 * dx * 2 ** 0.5 / 2
R_hole = 0.25

hole = geom.add_cylinder(0, 0, z0, 0, 0, h, R_hole)

# pillar = geom.add_box(-l_pillar / 2, -l_pillar / 2, z0, l_pillar, l_pillar, h)
# geom.rotate(pillar, (0, 0, 0), (0, 0, 1), np.pi / 4)
groove = geom.layers["groove"]
sub = geom.layers["substrate"]
sup = geom.layers["superstrate"]
sub, sup, hole, groove = geom.fragment([sub, sup, groove], hole)
geom.add_physical(hole, "hole")
geom.add_physical(groove, "groove")
geom.add_physical(sub, "substrate")
geom.add_physical(sup, "superstrate")
mesh_size = {d: lambda0 / param for d, param in mesh_param.items()}
geom.set_mesh_size(mesh_size)
# geom.remove_all_duplicates()
geom.build(interactive=0)
# geom.build(interactive=1)

######################################################################
# Set the permittivity and permeabilities for the various domains
# using a dictionary:

epsilon = {d: 1 for d in geom.domains}
mu = {d: 1 for d in geom.domains}
epsilon["groove"] = eps_layer
# epsilon["groove"] = eps_diel
epsilon["hole"] = 1
epsilon["substrate"] = eps_diel

######################################################################
# Now we can create an instance of the simulation class
# :class:`~gyptis.Grating`,

pw = gy.PlaneWave(lambda0, (theta0, phi0, psi0), dim=3, degree=degree)
grating = gy.Grating(
    geom, epsilon, mu, source=pw, degree=degree, periodic_map_tol=1e-12
)

# pp = gy.utils.project_iterative(pw.expression,grating.formulation.real_function_space)
# gy.dolfin.File("test.pvd") << pp.real

# us = grating.formulation.annex_field["as_subdomain"]["stack"]
# pp = gy.utils.project_iterative(us,grating.formulation.real_function_space)
#
# gy.dolfin.File("test.pvd") << pp.real
#
# import os
# os.system("paraview test.pvd")
# xsx

grating.solve()
N_d_order = 2
effs = grating.diffraction_efficiencies(N_d_order, orders=True)


print()
print("Transmission")
print("------------")
pprint.pprint(effs["T"])
print()
print("Reflection")
print("----------")
pprint.pprint(effs["R"])
print()
print("Absorption")
print("----------")
pprint.pprint(effs["Q"])
print()
print("Balance")
print("-------")
pprint.pprint(effs["B"])
print()

R00 = effs["R"][N_d_order][N_d_order]
print("R00")
print("-------")
print(R00)
print()
print("T")
print("-------")
print(np.sum(effs["T"]))
print()
print("R")
print("-------")
print(np.sum(effs["R"]))
print()
