#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.1
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
2D Anisotropic Grating
=======================

Example of diffraction grating with trapezoidal ridges made from an anisotropic material.
"""


from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import gyptis as gy

##############################################################################
# We will study this benchmark and compare with results
# given in :cite:p:`PopovGratingBook`.

fig, ax = plt.subplots(3, 2, figsize=(3.5, 5.5))


lambda0 = 633
period = 600

width_bottom, width_top = 500, 300
height = 600
eps_sub = 2.25
eps_rod = np.array([[2.592, 0.251, 0], [0.251, 2.592, 0], [0, 0, 2.829]])

pmesh = 10

thicknesses = OrderedDict(
    {
        "pml_bottom": 1 * lambda0,
        "substrate": 2 * lambda0,
        "groove": height * 1.5,
        "superstrate": 2 * lambda0,
        "pml_top": 1 * lambda0,
    }
)

mesh_param = dict(
    {
        "pml_bottom": pmesh * eps_sub**0.5,
        "substrate": pmesh * eps_sub**0.5,
        "groove": pmesh,
        "rod": pmesh * np.max(eps_rod) ** 0.5,
        "superstrate": pmesh,
        "pml_top": pmesh,
    }
)


geom = gy.Layered(2, period, thicknesses)
groove = geom.layers["groove"]
substrate = geom.layers["substrate"]
y0 = geom.y_position["groove"]
P = [geom.add_point(-width_bottom / 2, y0, 0)]
P.append(geom.add_point(width_bottom / 2, y0, 0))
P.append(geom.add_point(width_top / 2, y0 + height, 0))
P.append(geom.add_point(-width_top / 2, y0 + height, 0))
L = [
    geom.add_line(P[0], P[1]),
    geom.add_line(P[1], P[2]),
    geom.add_line(P[2], P[3]),
    geom.add_line(P[3], P[0]),
]
cl = geom.add_curve_loop(L)
rod = geom.add_plane_surface(geom.dimtag(cl, 1)[0])
substrate, groove, rod = geom.fragment([substrate, groove], rod)
geom.add_physical(rod, "rod")
geom.add_physical(groove, "groove")
geom.add_physical(substrate, "substrate")
mesh_size = {d: lambda0 / param for d, param in mesh_param.items()}
geom.set_mesh_size(mesh_size)

geom.build()
all_domains = geom.subdomains["surfaces"]
domains = [k for k in all_domains.keys() if k not in ["pml_bottom", "pml_top"]]

epsilon = {d: 1 for d in domains}
mu = {d: 1 for d in domains}

epsilon["substrate"] = eps_sub
epsilon["rod"] = eps_rod


nper = 8

for jangle, angle in enumerate([0, -20, -40]):
    angle_degree = angle * np.pi / 180

    pw = gy.PlaneWave(lambda0, angle_degree, dim=2)
    grating_TM = gy.Grating(geom, epsilon, mu, source=pw, polarization="TM", degree=2)
    grating_TM.solve()
    effs_TM = grating_TM.diffraction_efficiencies(2, orders=True)

    print(f"angle = {angle}, TM polarization")
    print("--------------------------------")
    for i in range(5):
        print(f"T {i-2}: {effs_TM["T"][i]:.6f}")
    for i in range(5):
        print(f"R {i-2}: {effs_TM["R"][i]:.6f}")
    B = sum(effs_TM["T"]) + sum(effs_TM["R"])
    print(f"B: {B:.6f}")

    ylim = geom.y_position["substrate"], geom.y_position["pml_top"]
    d = grating_TM.period
    vmin_TM, vmax_TM = -1.5, 1.7
    plt.sca(ax[jangle][0])
    per_plots, cb = grating_TM.plot_field(nper=nper)
    cb.remove()
    scatt_lines, layers_lines = grating_TM.plot_geometry(nper=nper, c="k")
    [layers_lines[i].remove() for i in [0, 1, 3, 4]]
    plt.ylim(ylim)
    plt.xlim(-d / 2, nper * d - d / 2)
    plt.axis("off")

    # TE

    grating_TE = gy.Grating(geom, epsilon, mu, source=pw, polarization="TE", degree=2)

    grating_TE.solve()
    effs_TE = grating_TE.diffraction_efficiencies(2, orders=True)

    H = grating_TE.solution["total"]
    print(f"angle = {angle}, TE polarization")
    print("--------------------------------")
    for i in range(5):
        print(f"T {i-2}: {effs_TE["T"][i]:.6f}")
    for i in range(5):
        print(f"R {i-2}: {effs_TE["R"][i]:.6f}")
    B = sum(effs_TE["T"]) + sum(effs_TE["R"])
    print(f"B: {B:.6f}")

    vmin_TE, vmax_TE = -2.5, 2.5
    plt.sca(ax[jangle][1])
    per_plots, cb = grating_TE.plot_field(nper=nper)
    cb.remove()
    scatt_lines, layers_lines = grating_TE.plot_geometry(nper=nper, c="k")
    [layers_lines[i].remove() for i in [0, 1, 3, 4]]
    plt.ylim(ylim)
    plt.xlim(-d / 2, nper * d - d / 2)
    plt.axis("off")

    ax[jangle][0].set_title(rf"$\theta = {angle}\degree$")
    ax[jangle][1].set_title(rf"$\theta = {angle}\degree$")


divider = make_axes_locatable(ax[0, 0])
cax = divider.new_vertical(size="5%", pad=0.5)
fig.add_axes(cax)
mTM = plt.cm.ScalarMappable(cmap="RdBu")
mTM.set_clim(vmin_TM, vmax_TM)

cbarTM = fig.colorbar(mTM, cax=cax, orientation="horizontal")
cax.set_title(r"${\rm Re}\, E_z$ (TM)")

divider = make_axes_locatable(ax[0, 1])
cax = divider.new_vertical(size="5%", pad=0.5)

mTE = plt.cm.ScalarMappable(cmap="RdBu")
mTE.set_clim(vmin_TE, vmax_TE)
fig.add_axes(cax)
cbarTE = fig.colorbar(mTE, cax=cax, orientation="horizontal")
cax.set_title(r"${\rm Re}\, H_z$ (TE)")

plt.tight_layout()
plt.subplots_adjust(wspace=-0.1, hspace=-0.3)
