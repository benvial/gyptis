#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.1
# License: MIT
# See the documentation at gyptis.gitlab.io

"""
High contrast homogenization
============================

Metamaterial with high index inclusions
"""

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy

##############################################################################
# Results are compared with :cite:p:`Felbacq2005`.

d = 1
a = 0.4 * d
eps_i = 200 - 5j
v = (d, 0), (0, 9 * d)
lmin = a / 40

lattice = gy.Lattice(dim=2, vectors=((d, 0), (0, d)))
cell = lattice.cell
rod = lattice.add_circle(d / 2, 0.5 * d, 0, a)
rod, cell = lattice.fragment(cell, rod)
lattice.add_physical(cell, "background")
lattice.add_physical(rod, "inclusion")
lattice.set_size("background", lmin)
lattice.set_size("inclusion", lmin)
lattice.build()

epsilon = dict(inclusion=-1.0 - 1e-3j, background=1)
mu = dict(inclusion=1, background=1)
hom = gy.Homogenization2D(lattice, epsilon, mu, degree=2)

eps_eff = hom.get_effective_permittivity()
print(eps_eff)
plt.close("all")
v = hom.solution["epsilon"]["x"]
gy.plot(v)
plt.show()
v = hom.solution["epsilon"]["y"]
gy.plot(v)
plt.show()


##############################################################################
# Build the lattice


def make_rods(geom):
    rods = [geom.add_circle(d / 2, (i + 0.5) * d, 0, a) for i in range(9) if i != 4]
    rod_center = geom.add_circle(d / 2, 4.5 * d, 0, a)
    return geom, rods, rod_center


lattice = gy.Lattice(dim=2, vectors=v)
cell = lattice.cell
lattice, rods, rod_center = make_rods(lattice)
# cell = lattice.cut(lattice.cell, rods)
cell = lattice.cut(cell, rod_center)
*rods, cell = lattice.fragment(cell, rods)
lattice.add_physical(cell, "background")
lattice.add_physical(rods, "inclusion_out")
# lattice.add_physical(rod_center, "inclusion_center")
lattice.set_size("background", lmin)
lattice.set_size("inclusion_out", lmin)
lattice.build()


##############################################################################
# Build the inclusion

inclusion = gy.Geometry(dim=2)
inclusion, rods, rod_center = make_rods(inclusion)
inclusion.add_physical(rods, "inclusion")
inclusion.add_physical(rod_center, "inclusion_center")
bnds = inclusion.get_boundaries("inclusion")
bnds += inclusion.get_boundaries("inclusion_center")
inclusion.add_physical(bnds, "inclusion_bnds", dim=1)
inclusion.set_size("inclusion", lmin)
inclusion.set_size("inclusion_center", lmin)
inclusion.build()

##############################################################################
# Materials

epsilon = dict(inclusion_out=-1, inclusion=eps_i, background=1)
mu = dict(inclusion_out=1, inclusion=1, background=1)

##############################################################################
# Homogenization model

hom = gy.models.HighContrastHomogenization2D(lattice, inclusion, epsilon, mu, degree=2)

##############################################################################
# Effective permittivity

eps_eff = hom.get_effective_permittivity()
print(eps_eff)

##############################################################################
# Slab and negative refraction

wavelength = 9 * d
pmesh = 3

Nx = 10
Ny = 9
lbox_x = Nx * d + 1 * wavelength
lbox_y = Ny * d + 4 * wavelength


lmin = wavelength / pmesh
geom = gy.BoxPML(
    dim=2,
    box_size=(lbox_x, lbox_y),
    box_center=(0, 0),
    pml_width=(wavelength, wavelength),
)
box = geom.box
rods = [
    geom.add_square((-Nx / 2 + i) * d, (-Ny / 2 + j) * d, 0, a)
    for i in range(Nx)
    for j in range(Ny)
    if j != 3
]
*rods, box = geom.fragment(box, rods)
rods_central = [
    geom.add_square((-Nx / 2 + i) * d, (-Ny / 2 + i) * d, 0, a) for i in range(Nx)
]
*rods_central, box = geom.fragment(box, rods_central)
geom.add_physical(box, "box")
geom.add_physical(rods, "rods")
geom.add_physical(rods_central, "rods_central")
[geom.set_size(pml, lmin * 0.7) for pml in geom.pmls]
geom.set_size("box", lmin)
geom.set_size("rods_central", lmin / eps_i.real**0.5)
geom.set_size("rods", lmin)
geom.build(1)


pw = gy.GaussianBeam(
    wavelength=wavelength,
    angle=3 * gy.pi / 4,
    waist=wavelength,
    position=(0, 0),
    dim=2,
    domain=geom.mesh,
    degree=2,
)

epsilon = dict(box=1, rods=eps_i, rods_central=-1)
mu = dict(box=1, rods=1, rods_central=1)
s = gy.Scattering(
    geom,
    epsilon,
    mu,
    pw,
    degree=2,
    polarization="TE",
)

s.solve()
plt.figure(figsize=(4, 1.8))
s.plot_field()
ax = plt.gca()
# for i in range(N):
#     cir = plt.Circle((-N / 2 * d + i * d, 0), R, lw=0.3, color="w", fill=False)
#     ax.add_patch(cir)
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
# plt.title("title")
plt.tight_layout()
