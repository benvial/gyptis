#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.1
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
2D Dielectric Grating
=====================

Example of a dielectric diffraction grating.
"""
# sphinx_gallery_thumbnail_number = 2


from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy

plt.close("all")

plt.ion()

period = 1.8  # period of the grating
lambda0 = 1
h = lambda0
theta0 = 0  # in degrees
eps_diel = 4
l_pillar = period / 2

##############################################################################
# The thicknesses of the different layers are specified with an
# ``OrderedDict`` object **from bottom to top**:

thicknesses = OrderedDict(
    {
        "pml_bottom": lambda0,
        "substrate": lambda0 / 1,
        "groove": h,
        "superstrate": lambda0 / 1,
        "pml_top": lambda0,
    }
)

##############################################################################
# Here we set the mesh refinement parameters, in order to be able to have
# ``parmesh`` cells per wavelength of the field inside each subdomain
degree = 2
polarization = "TE"
pmesh = 6
pmesh_pillar = pmesh * 1
pmesh_groove = pmesh * 1
mesh_param = dict(
    {
        "pml_bottom": 1 * pmesh * eps_diel**0.5,
        "substrate": pmesh * eps_diel**0.5,
        "groove": pmesh_groove * eps_diel**0.5,
        "rod": pmesh_pillar * eps_diel**0.5,
        "superstrate": pmesh,
        "pml_top": 1 * pmesh,
    }
)


##############################################################################
# Let's create the geometry using the :class:`~gyptis.Layered`
# class:
geom = gy.Layered(2, dy, thicknesses)

y0 = geom.y_position["groove"]
rod = geom.add_rectangle(-l_pillar / 2, y0, 0, l_pillar, h)

groove = geom.layers["groove"]
sub = geom.layers["substrate"]
sup = geom.layers["superstrate"]
out = geom.fragment([sub, sup, groove], rod)
sub = out[0]
sup = out[1]
rod = out[2]
groove = out[3:]
geom.add_physical(rod, "rod")
geom.add_physical(groove, "groove")
geom.add_physical(sub, "substrate")
geom.add_physical(sup, "superstrate")
mesh_size = {d: lambda0 / param for d, param in mesh_param.items()}
geom.set_mesh_size(mesh_size)
geom.build()

######################################################################
# Set the permittivity and permeabilities for the various domains
# using a dictionary:

domains = geom.subdomains["surfaces"]
epsilon = {d: 1 for d in domains}
# epsilon["rod"] = n_rod**2
epsilon["rod"] = eps_diel
# epsilon["rod"] = epsilon["groove"] = eps_diel
epsilon["substrate"] = eps_diel
mu = {d: 1 for d in domains}

######################################################################
# Now we can create an instance of the simulation class
# :class:`~gyptis.Grating`, where we specify the
# Transverse Electric polarization case (electric field out of plane
# :math:`\boldsymbol{E} = E_z \boldsymbol{e_z}`) and the ``degree`` of
# Lagrange finite elements.

angle = theta0 * gy.pi / 180

pw = gy.PlaneWave(lambda0, angle, dim=2)

gratingTE = gy.Grating(
    geom, epsilon, mu, source=pw, polarization=polarization, degree=degree
)
stk = gratingTE.formulation.annex_field["stack_output"]
# print(stk)

gratingTE.solve()
nord = 3
effs_TE = gratingTE.diffraction_efficiencies(nord, orders=True)
E = gratingTE.solution["total"]

# reference
T_ref = dict(TM=[0.2070, 1.0001], TE=[0.8187, 1.0001])
print()
print("     case        T         R       B")
print("------------------------------------------------")
print(
    f"   grating     {effs_TE['T'][nord]:.4f}   {effs_TE['R'][nord]:.4f}    {effs_TE['B']:.4f}"
)
print(
    f"   stack       {stk[2]['T'].real:.4f}   {stk[2]['R'].real:.4f}    {stk[2]['R'].real+ stk[2]['T'].real + stk[2]['Q'].real:.4f}"
)

print(effs_TE)

utot = gratingTE.solution["total"]
gy.plot(utot)
# u = gratingTE.solution["periodic"]
# gy.plot(u)


# ######################################################################
# # We switch to TE polarization

# gratingTM = gy.Grating(geom, epsilon, mu, source=pw, polarization="TE", degree=2)
# gratingTM.solve()
# effs_TM = gratingTM.diffraction_efficiencies(1, orders=True)
# H = gratingTM.solution["total"]


# ######################################################################
# # Let's visualize the fields

# fig, ax = plt.subplots(1, 2)
# ylim = geom.y_position["substrate"], geom.y_position["pml_top"]
# gratingTE.plot_field(ax=ax[0])
# gratingTE.plot_geometry(ax=ax[0])
# ax[0].set_ylim(ylim)
# ax[0].set_axis_off()
# ax[0].set_title("$E_z$ (TM)")
# gratingTM.plot_field(ax=ax[1])
# gratingTM.plot_geometry(ax=ax[1])
# ax[1].set_ylim(ylim)
# ax[1].set_axis_off()
# ax[1].set_title("$H_z$ (TE)")
# fig.tight_layout()
# fig.show()

# ######################################################################
# # Results are in good agreement with the reference

# print("Transmission coefficient")
# print(" order      ref       calc")
# print("--------------------------------")
# print(f"   0       {T_ref['TE'][0]:.4f}    {effs_TM['T'][1]:.4f} ")
# print(f"  sum      {T_ref['TE'][1]:.4f}    {effs_TM['B']:.4f}   ")
