#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.1
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
Lossy Torus Grating
===================

Crossed grating with toroidal inclusions.
"""
# sphinx_gallery_thumbnail_number = 2

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy

gy.dolfin.parameters["form_compiler"]["quadrature_degree"] = 5
# gy.dolfin.parameters["ghost_mode"] = "shared_facet"
# gy.dolfin.set_log_level(7)


ref0 = {"R": 0.36376, "T": 0.32992, "Q": 0.30639}
ref40 = {"R": 0.27331, "T": 0.38191, "Q": 0.34476}

##############################################################################
# Structure is the same as in :cite:p:`Demesy2010`.
#
# The units of lengths are in nanometers here, and we first define some
# geometrical and optical parameters:


lambda0 = 1
dx = dy = 0.3  # 5 * lambda0 * 2 ** 0.5 / 4  # periods of the grating
a = 0.1 / 2
b = 0.05 / 2
R = 0.15 / 2
theta0 = 0 * gy.pi / 180
phi0 = 0
psi0 = 0
eps_diel = 2.25
eps_incl = -21 - 20j

h = 2 * b

##############################################################################
# The thicknesses of the different layers are specified with an
# ``OrderedDict`` object **from bottom to top**:

h_supsub = lambda0 / 0.5

thicknesses = OrderedDict(
    {
        "pml_bottom": lambda0,
        "substrate": h_supsub,
        "groove": h,
        "superstrate": h_supsub,
        "pml_top": lambda0,
    }
)

##############################################################################
# Here we set the mesh refinement parameters, in order to be able to have
# ``parmesh`` cells per wavelength of the field inside each subdomain

degree = 2
pmesh = 20
pmesh_torus = pmesh * 1.0
mesh_param = dict(
    {
        "pml_bottom": 1 * pmesh * eps_diel**0.5,
        "substrate": pmesh * eps_diel**0.5,
        "groove": pmesh,
        "torus": pmesh_torus * abs(eps_incl) ** 0.5,
        "superstrate": pmesh,
        "pml_top": 1 * pmesh,
    }
)

##############################################################################
# Let's create the geometry using the :class:`~gyptis.Layered`
# class:
geom = gy.Layered(3, (dx, dy), thicknesses)
z0 = geom.z_position["groove"]
torus = geom.add_torus(0, 0, z0 + h / 2, R, a)
geom.dilate(geom.dimtag(torus), 0, 0, z0 + h / 2, 1, 1, b / a)
groove = geom.layers["groove"]
sub = geom.layers["substrate"]
sup = geom.layers["superstrate"]
sub, sup, torus, *groove = geom.fragment([sub, sup, groove], torus)
geom.add_physical(torus, "torus")
geom.add_physical(groove, "groove")
geom.add_physical(sub, "substrate")
geom.add_physical(sup, "superstrate")
mesh_size = {d: lambda0 / param for d, param in mesh_param.items()}
geom.set_mesh_size(mesh_size)
geom.build(interactive=1)
# geom.build(interactive=1)

xsxs


######################################################################
# Set the permittivity and permeabilities for the various domains
# using a dictionary:

epsilon = {d: 1 for d in geom.domains}
mu = {d: 1 for d in geom.domains}
epsilon["torus"] = eps_incl
epsilon["substrate"] = eps_diel

######################################################################
# Now we can create an instance of the simulation class
# :class:`~gyptis.Grating`,

pw = gy.PlaneWave(lambda0, (theta0, phi0, psi0), dim=3, degree=degree)
grating = gy.Grating(
    geom,
    epsilon,
    mu,
    source=pw,
    degree=degree,
    periodic_map_tol=1e-11,
    eps_bc=1e-11,
)

grating.solve()
effs = grating.diffraction_efficiencies(0, orders=True)

mprint = gy.utils.mpi_print
mprint(effs)

errR = abs(1 - effs["R"][0][0] / ref0["R"])
errT = abs(1 - effs["T"][0][0] / ref0["T"])
errQ = abs(1 - effs["Q"] / ref0["Q"])
mprint("errors")
mprint("R: ", errR)
mprint("T: ", errT)
mprint("Q: ", errQ)
