#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
3D Checkerboard Grating
=======================

Example of a dielectric bi-periodic diffraction grating.
"""
# sphinx_gallery_thumbnail_number = 2


from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy
import time

gy.dolfin.parameters["form_compiler"]["quadrature_degree"] = 5
# gy.dolfin.parameters["ghost_mode"] = "shared_facet"
gy.dolfin.set_log_level(100)

##############################################################################
# Structure is the same as in :cite:p:`Demesy2010`.
#
# The units of lengths are in nanometers here, and we first define some
# geometrical and optical parameters:


lambda0 = 1
dx = dy = 5 * lambda0 * 2**0.5 / 4  # periods of the grating
h = lambda0
theta0 = 0
phi0 = 0
psi0 = gy.pi / 4
eps_diel = 2.25
eps_layer = 1

##############################################################################
# The thicknesses of the different layers are specified with an
# ``OrderedDict`` object **from bottom to top**:

thicknesses = OrderedDict(
    {
        "pml_bottom": lambda0,
        "substrate": lambda0  * 0.5,
        "substrate1": lambda0  * 0.5,
        "groove": h,
        "superstrate1": lambda0 * 0.1,
        "superstrate": lambda0 * 0.1,
        "pml_top": lambda0,
    }
)

##############################################################################
# Here we set the mesh refinement parameters, in order to be able to have
# ``parmesh`` cells per wavelength of the field inside each subdomain

degree = 2
pmesh = 3
pmesh_pillar = pmesh * 1
pmesh_groove = pmesh * 1
mesh_param = dict(
    {
        "pml_bottom": 1 * pmesh * eps_diel**0.5,
        "substrate": pmesh * eps_diel**0.5,
        "substrate1": pmesh * eps_diel**0.5,
        "groove": pmesh_groove * eps_layer**0.5,
        "pillar": pmesh_pillar * eps_diel**0.5,
        "superstrate1": pmesh,
        "superstrate": pmesh,
        "pml_top": 1 * pmesh,
    }
)

##############################################################################
# Let's create the geometry using the :class:`~gyptis.Layered`
# class:
geom = gy.Layered(3, (dx, dy), thicknesses)
z0 = geom.z_position["groove"]  # + h/10
l_pillar = dx * 2**0.5 / 2
pillar = geom.add_box(-l_pillar / 2, -l_pillar / 2, z0, l_pillar, l_pillar, h)
geom.rotate(pillar, (0, 0, 0), (0, 0, 1), np.pi / 4)
groove = geom.layers["groove"]
sub = geom.layers["substrate1"]
sup = geom.layers["superstrate1"]
# sub, sup, pillar, groove = geom.fragment([sub, sup, groove], pillar)
out = geom.fragment([sub, sup, groove], pillar)
sub = out[0]
sup = out[1]
pillar = out[2]
groove = out[3:]
geom.add_physical(pillar, "pillar")
geom.add_physical(groove, "groove")
geom.add_physical(sub, "substrate1")
geom.add_physical(sup, "superstrate1")
sub = geom.layers["substrate1"]
sup = geom.layers["superstrate1"]
# geom.add_physical(sub, "substrate")
# geom.add_physical(sup, "superstrate")
mesh_size = {d: lambda0 / param for d, param in mesh_param.items()}
geom.set_mesh_size(mesh_size)
# geom.remove_all_duplicates()
geom.build(interactive=False)
# geom.build(interactive=True)

# import sys
# sys.exit(0)

######################################################################
# Set the permittivity and permeabilities for the various domains
# using a dictionary:

mu = {d: 1 for d in geom.domains}
epsilon = {d: 1 for d in geom.domains}
epsilon["pillar"] = 1
epsilon["superstrate"] = 1
epsilon["superstrate1"] = 1
# epsilon["substrate1"] = eps_diel
epsilon["substrate"] = eps_diel

######################################################################
# Now we can create an instance of the simulation class
# :class:`~gyptis.Grating`,

pw = gy.PlaneWave(lambda0, (theta0, phi0, psi0), dim=3, degree=degree)
grating = gy.Grating(
    geom, epsilon, mu, source=pw, degree=degree, periodic_map_tol=1e-12
)
print(grating.formulation.annex_field["stack_output"])

# epsilon1 = {d: 1 for d in geom.domains}
# epsilon1["superstrate"] = eps_diel
# epsilon1["substrate"] = 1

# pw1= gy.PlaneWave(lambda0, (theta0, phi0, psi0), dim=3, degree=degree)
# grating = gy.Grating(
#     geom, epsilon1, mu, source=pw1, degree=degree, periodic_map_tol=1e-12
# )
# print(grating.formulation.annex_field["stack_output"])

# xsx


# pp = gy.utils.project_iterative(pw.expression,grating.formulation.real_function_space)
# gy.dolfin.File("test.pvd") << pp.real

# us = grating.formulation.annex_field["as_subdomain"]["stack"]
# pp = gy.utils.project_iterative(us,grating.formulation.real_function_space)

# gy.dolfin.File("test.pvd") << pp.real

# import os
# os.system("paraview test.pvd")

t = -time.time()

grating.solve()


V = gy.dolfin.FunctionSpace(geom.mesh,"CG",2)

reEx = grating.solution["total"][0].real
pp = gy.project_iterative(reEx,V)
gy.dolfin.File("reEx.pvd") << pp
import os
os.system("paraview reEx.pvd")

N = 1
effs = grating.diffraction_efficiencies(N, orders=True)
# print(effs)
t += time.time()

Tfmm = [
    0.04308,
    0.12860,
    0.06196,
    0.12860,
    0.17486,
    0.12860,
    0.06196,
    0.12860,
    0.04308,
]
Tfem = [
    0.04333,
    0.12845,
    0.06176,
    0.12838,
    0.17577,
    0.12839,
    0.06177,
    0.12843,
    0.04332,
]

Tgyptis = np.array(effs["T"]).ravel()
sumRgyptis = np.sum(effs["R"])
Bgyptis = effs["B"]


k = 0
print()
print("                 FMM          FEM         gyptis      ")
print("------------------------------------------------------")
for i in range(-N, N + 1):
    i1 = f"+{i}" if i > 0 else i
    spacex = "" if i != 0 else " "
    for j in range(-N, N + 1):
        j1 = f"+{j}" if j > 0 else j
        spacey = "" if j != 0 else " "
        print(
            f"T({i1},{j1}) {spacex}{spacey}      {Tfmm[k]:.5f}      {Tfem[k]:.5f}      {Tgyptis[k]:.5f}"
        )
        k += 1
print()
print(f"sum R            --         {0.10040:.5f}      {sumRgyptis:.5f}")
print()
print(f"TOTAL            --         {1.00000:.5f}      {Bgyptis:.5f}")
print()
print(f"CPU time         --            --        {t:.1f}s")
