#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.3
# License: MIT
# See the documentation at gyptis.gitlab.io

"""
Nanorod
==================

Scattering by a silver nanorod and comparison with analytical solution.
"""


import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy
import gyptis.utils.data_download as dd
from gyptis import c, pi
from gyptis.utils import adaptive_sampler

##############################################################################
# Create the geometry and mesh

pmesh = 10
degree = 2
wavelength = 0.452
R = 0.50
eps_rod = -6.15 - 0.73j  # silver @452nm


Rcalc = 1.2 * R
lbox = Rcalc * 2 * 1.1
pml_width = wavelength

geom = gy.BoxPML(
    dim=2,
    box_size=(lbox, lbox),
    pml_width=(pml_width, pml_width),
    Rcalc=Rcalc,
)
box = geom.box
cyl = geom.add_circle(0, 0, 0, R)
out = geom.fragment(cyl, box)
box = out[1:3]
cyl = out[0]
geom.add_physical(box, "box")
geom.add_physical(cyl, "rod")

lmin = wavelength / pmesh
[geom.set_size(pml, lmin * 0.75) for pml in geom.pmls]
geom.set_size("box", lmin)
n_rod = abs(eps_rod.real) ** 0.5
geom.set_size("rod", 0.5 * lmin / n_rod)
geom.build()


##############################################################################
# Define the incident plane wave and materials

pw = gy.PlaneWave(
    wavelength=wavelength, angle=0, dim=2, domain=geom.mesh, degree=degree
)
epsilon = dict(box=1, rod=eps_rod)


##############################################################################
# Scattering problem

s = gy.Scattering(
    geom,
    epsilon,
    source=pw,
    degree=degree,
    polarization="TE",
)
s.solve()
s.plot_field()
geom_lines = geom.plot_subdomains()
plt.xlabel(r"$x$ (nm)")
plt.ylabel(r"$y$ (nm)")
plt.title(r"Re $H_z$")
plt.tight_layout()


##############################################################################
# Compute cross sections and check energy conservation (optical theorem)

cs = s.get_cross_sections()

print("Gyptis")
print("--------")
print("scattering cross section: ", cs["scattering"])
print("absorption cross section: ", cs["absorption"])
print("extinction cross section: ", cs["extinction"])
print("optical theorem: ", abs(cs["scattering"] + cs["absorption"] - cs["extinction"]))
assert np.allclose(cs["extinction"], cs["scattering"] + cs["absorption"], rtol=5e-4)


##############################################################################
# Compare with analytical solution

from effs_scatt_cylinder import calculate_analytical_cross_sections

cs_ana = calculate_analytical_cross_sections(eps_rod, 1, wavelength, R, N=50)

print()
print("Analytical")
print("----------")
print("scattering cross section: ", cs_ana["scattering"])
print("absorption cross section: ", cs_ana["absorption"])
print("extinction cross section: ", cs_ana["extinction"])

err_scattering = abs(cs_ana["scattering"] - cs["scattering"]) / cs["scattering"]
err_absorption = abs(cs_ana["absorption"] - cs["absorption"]) / cs["absorption"]
err_extinction = abs(cs_ana["extinction"] - cs["extinction"]) / cs["extinction"]

print()
print("Errors")
print("----------")
print(f"scattering:  {100*err_scattering} %")
print(f"absorption: {100*err_absorption} %")
print(f"extinction: {100*err_extinction} %")
assert err_absorption < 0.01
assert err_scattering < 0.01
assert err_extinction < 0.01
