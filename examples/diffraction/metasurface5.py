#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.2
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
Metasurface
========================

With multiple layers.
"""


import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pyvista

import gyptis as gy

##############################################################################
# We first define some parameters

lambda0 = 1
dy = 1.8
dx = dy
h = lambda0 / 6
theta0 = 0
phi0 = 0
eps_diel = 4
eps_pillar = 3 - 0.2j
eps_layer = 1
degree = 2
pmesh = 3
nord = 3


gy.dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

##############################################################################
# The thicknesses of the different layers are specified with an
# ``OrderedDict`` object **from bottom to top**:

thicknesses = OrderedDict(
    {
        "pml_bottom": lambda0,
        "substrate": lambda0,
        "layer1": h,
        "layer2": h,
        "layer3": h,
        "layer4": h,
        "layer5": h,
        "layer6": h,
        "superstrate": lambda0,
        "pml_top": lambda0,
    }
)

##############################################################################
# Here we set the mesh refinement parameters, in order to be able to have
# ``parmesh`` cells per wavelength of the field inside each subdomain

mesh_param = dict(
    {
        "pml_bottom": 0.7 * pmesh * eps_diel**0.5,
        "substrate": pmesh * eps_diel**0.5,
        "layer1": pmesh * eps_layer**0.5,
        "layer2": pmesh * eps_layer**0.5,
        "layer3": pmesh * eps_layer**0.5,
        "layer4": pmesh * eps_layer**0.5,
        "layer5": pmesh * eps_layer**0.5,
        "layer6": pmesh * eps_layer**0.5,
        "pillar1": pmesh * abs(eps_pillar.real) ** 0.5,
        "pillar2": pmesh * abs(eps_pillar.real) ** 0.5,
        "pillar3": pmesh * abs(eps_pillar.real) ** 0.5,
        "pillar4": pmesh * abs(eps_pillar.real) ** 0.5,
        "pillar5": pmesh * abs(eps_pillar.real) ** 0.5,
        "pillar6": pmesh * abs(eps_pillar.real) ** 0.5,
        "superstrate": pmesh,
        "pml_top": 0.7 * pmesh,
    }
)

##############################################################################
# Let's create the geometry using the :class:`~gyptis.Layered`
# class:


geom = gy.Layered(3, (dx, dy), thicknesses)
sub = geom.layers["substrate"]
sup = geom.layers["superstrate"]
pillars = []
layers = []
for i in range(1, 7):
    l_pillar = dy / (i + 3)
    z0 = geom.z_position[f"layer{i}"]
    pillar = geom.add_box(-l_pillar / 2, -l_pillar / 2, z0, l_pillar, l_pillar, h)

    layer = geom.layers[f"layer{i}"]
    out = geom.fragment([layer], pillar)
    pillar = out[0]
    layer = out[1]
    pillars.append(pillar)
    layers.append(layer)
mesh_size = {d: lambda0 / param for d, param in mesh_param.items()}
# geom.add_physical(sub, "substrate")
# geom.add_physical(sup, "superstrate")
geom.remove_all_duplicates()
vols = geom.get_entities(dim=3)
import gmsh

gmsh.model.removePhysicalGroups()
geom.add_physical(vols[0][1], "pml_bottom")
geom.add_physical(vols[1][1], "substrate")
geom.add_physical(vols[2][1], "superstrate")
geom.add_physical(vols[3][1], "pml_top")
for i in range(1, 7):
    geom.add_physical(vols[2 * i + 2][1], f"pillar{i}")
    geom.add_physical(vols[2 * i + 3][1], f"layer{i}")

# geom.add_physical(vols[-2][1], "superstrate")
# geom.add_physical(vols[-1][1], "pml_top")
geom.set_mesh_size(mesh_size)

t3Dmesh = -time.time()
geom.build(interactive=0, read_mesh=True, read_info=True)
t3Dmesh += time.time()


######################################################################
# Now we can create an instance of the simulation class
# :class:`~gyptis.Grating`,

polarization = "TM"
mu = {d: 1 for d in geom.domains}
epsilon = {d: 1 for d in geom.domains}
for i in range(1, 6):
    epsilon[f"pillar{i}"] = eps_pillar
    epsilon[f"layer{i}"] = eps_layer
epsilon["substrate"] = eps_diel

psi0 = 0 if polarization == "TM" else gy.pi / 2
pw = gy.PlaneWave(lambda0, (theta0, phi0, psi0), dim=3, degree=degree)
simu3D = gy.Grating(geom, epsilon, mu, source=pw, degree=degree, periodic_map_tol=1e-12)


t3Dsolve = -time.time()
simu3D.solve()
t3Dsolve += time.time()

t3Deff = -time.time()
effs3D = simu3D.diffraction_efficiencies(nord, orders=True)
t3Deff += time.time()

T3D = np.array(effs3D["T"])[nord]
R3D = np.array(effs3D["R"])[nord]
Q3D = effs3D["Q"]
B3D = effs3D["B"]


print()
print("=========================================")
print(f"             {polarization} polarization")
print("=========================================")
print()

print("Transmission")
for k, i in enumerate(range(-nord, nord + 1)):
    i1 = f"+{i}" if i > 0 else i
    spacex = "" if i != 0 else " "
    print(f"T({i1}) {spacex}                  {T3D[k]:.5f}  ")
print(f"T                       {sum(T3D):.5f}  ")
print()
print("Reflection")
for k, i in enumerate(range(-1, 2), start=2):
    i1 = f"+{i}" if i > 0 else i
    spacex = "" if i != 0 else " "
    print(f"R({i1}) {spacex}                  {R3D[k]:.5f}  ")
print(f"R                       {sum(R3D):.5f}  ")
print()
print("Absorption")
print(f"Q                       {Q3D:.5f}  ")
print()
print("Energy balance")
print(f"B                       {B3D:.5f}  ")
print()
print("Number of DOF")
print(f"                        {simu3D.ndof}  ")
print()
print("CPU time")
print(f"mesh                       {t3Dmesh:.3f}s  ")
print(f"solve                      {t3Dsolve:.3f}s  ")
print(f"efficiencies               {t3Deff:.3f}s  ")


V = gy.dolfin.FunctionSpace(geom.mesh, "CG", 1)
E = simu3D.solution["total"]
H = simu3D.formulation.get_magnetic_field(E)
comp = 0
Z0 = (gy.mu_0 / gy.epsilon_0) ** 0.5
field = Z0 * H if polarization == "TE" else E
gy.dolfin.File("reE.pvd") << gy.project_iterative(field[comp].real, V)
reader = pyvista.get_reader("reE.pvd")
mesh = reader.read()
pl = pyvista.Plotter()
_ = pl.add_mesh(
    mesh,
    cmap="RdBu_r",
    interpolate_before_map=True,
    scalar_bar_args={"title": "Re E", "vertical": True},
)
pl.add_title(f"3D, {polarization}")
pl.view_yz()
pl.show()
