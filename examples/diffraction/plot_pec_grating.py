# -*- coding: utf-8 -*-
"""
2D PEC Grating
==============

Example of a perfectly conducting diffraction grating.
"""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy

##############################################################################
# We will study a classical benchmark of a perfectly conducting
# grating and compare with results given in :cite:p:`PopovGratingBook`.
#

lambda0 = 600
theta0 = 20

period = 800
h = 8
w = 600

pmesh = 10

thicknesses = OrderedDict(
    {
        "pml_bottom": 1 * lambda0,
        "substrate": 1 * lambda0,
        "groove": 20 * h,
        "superstrate": 1 * lambda0,
        "pml_top": 1 * lambda0,
    }
)

mesh_param = dict(
    {
        "pml_bottom": 0.7 * pmesh,
        "substrate": pmesh,
        "groove": pmesh,
        "superstrate": pmesh,
        "pml_top": 0.7 * pmesh,
    }
)

######################################################################
# Let's create the geometry

geom = gy.Layered(2, period, thicknesses)
groove = geom.layers["groove"]
y0 = geom.y_position["groove"] + thicknesses["groove"] / 2
rod = geom.add_ellipse(0, y0, 0, w / 2, h / 2)
groove = geom.cut(groove, rod, removeTool=True)
geom.add_physical(groove, "groove")

mesh_size = {d: lambda0 / param for d, param in mesh_param.items()}


rod_bnds = geom.get_boundaries("groove")[-1]
geom.add_physical(rod_bnds, "rod_bnds", dim=1)
geom.set_mesh_size(mesh_size)
geom.set_mesh_size({"rod_bnds": h / 8}, dim=1)

geom.build(
    interactive=False,
    generate_mesh=True,
    write_mesh=True,
    read_info=True,
)


domains = [k for k in thicknesses.keys() if k not in ["pml_bottom", "pml_top"]]

epsilon = {d: 1 for d in domains}
mu = {d: 1 for d in domains}

angle = theta0 * np.pi / 180
pw = gy.PlaneWave(lambda0, angle, dim=2)

gratingTM = gy.Grating(
    geom,
    epsilon,
    mu,
    source=pw,
    polarization="TM",
    degree=2,
    boundary_conditions={"rod_bnds": "PEC"},
)


gratingTM.solve()
effs_TM = gratingTM.diffraction_efficiencies(1, orders=True)

E = gratingTM.solution["total"]

### reference
T_ref = dict(TM=[0.0639, 1.0000], TE=[0.1119, 1.0000])

print("Transmission coefficient (TM)")
print(" order      ref       calc")
print("--------------------------------")
print(f"   0       {T_ref['TM'][0]:.4f}    {effs_TM['T'][1]:.4f} ")
print(f"  sum      {T_ref['TM'][1]:.4f}    {effs_TM['B']:.4f}   ")


######################################################################
# We switch to TE polarization

gratingTE = gy.Grating(
    geom,
    epsilon,
    mu,
    source=pw,
    polarization="TE",
    degree=2,
    boundary_conditions={"rod_bnds": "PEC"},
)

gratingTE.solve()
effs_TE = gratingTE.diffraction_efficiencies(2, orders=True)

H = gratingTE.solution["total"]


fig, ax = plt.subplots(1, 2)
ylim = geom.y_position["substrate"], geom.y_position["pml_top"]
gratingTM.plot_field(ax=ax[0])
gratingTM.plot_geometry(ax=ax[0])
ax[0].set_ylim(ylim)
ax[0].set_axis_off()
ax[0].set_title("$E_z$ (TM)")
gratingTE.plot_field(ax=ax[1])
gratingTE.plot_geometry(ax=ax[1])
ax[1].set_ylim(ylim)
ax[1].set_axis_off()
ax[1].set_title("$H_z$ (TE)")
fig.tight_layout()
fig.show()

print("Transmission coefficient (TE)")
print(" order      ref       calc")
print("--------------------------------")
print(f"   0       {T_ref['TE'][0]:.4f}    {effs_TE['T'][1]:.4f} ")
print(f"  sum      {T_ref['TE'][1]:.4f}    {effs_TE['B']:.4f}   ")
