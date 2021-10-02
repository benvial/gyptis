# -*- coding: utf-8 -*-
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

##############################################################################
# We will study a classical benchmark of a dielectric grating
# and compare with results given in :cite:p:`PopovGratingBook`.
#
# The units of lengths are in nanometers here, and we first define some
# geometrical and optical parameters:

period = 800  # period of the grating
ax, ay = 300, 200  # semi axes of the elliptical rods along x and y
n_rod = 1.4  # refractive index of the rods

##############################################################################
# The grating is illuminated from the top by a plane wave of wavelength
# ``lambda0`` and angle ``theta0`` with respect to the normal to the
# interface (the :math:`y` axis)

lambda0 = 600
theta0 = 20  # in degrees

##############################################################################
# The thicknesses of the different layers are specified with an
# ``OrderedDict`` object **from bottom to top**:

thicknesses = OrderedDict(
    {
        "pml_bottom": lambda0,
        "substrate": lambda0,
        "groove": 2 * ay * 1.5,
        "superstrate": lambda0,
        "pml_top": lambda0,
    }
)

##############################################################################
# Here we set the mesh refinement parameters, in order to be able to have
# ``parmesh`` cells per wavelength of the field inide each media

pmesh = 10
pmesh_rod = pmesh * 2
mesh_param = dict(
    {
        "pml_bottom": 0.7 * pmesh,
        "substrate": pmesh,
        "groove": pmesh,
        "rod": pmesh_rod * n_rod,
        "superstrate": pmesh,
        "pml_top": 0.7 * pmesh,
    }
)

##############################################################################
# Let's create the geometry using the :class:`~gyptis.Layered`
# class:
geom = gy.Layered(2, period, thicknesses)
groove = geom.layers["groove"]
y0 = geom.y_position["groove"] + thicknesses["groove"] / 2
rod = geom.add_ellipse(0, y0, 0, ax, ay)
rod, groove = geom.fragment(groove, rod)
geom.add_physical(rod, "rod")
geom.add_physical(groove, "groove")
mesh_size = {d: lambda0 / param for d, param in mesh_param.items()}
geom.set_mesh_size(mesh_size)
geom.build()


######################################################################
# The mesh can be visualized:

plt.figure()
geom.plot_mesh(lw=1)
geom.plot_subdomains(lw=2, c="#d76c4a")
plt.axis("off")
plt.tight_layout()
plt.show()


######################################################################
# Set the permittivity and permeabilities for the various domains
# using a dictionary:

domains = geom.subdomains["surfaces"]
epsilon = {d: 1 for d in domains}
epsilon["rod"] = n_rod ** 2
mu = {d: 1 for d in domains}

######################################################################
# Now we can create an instance of the simulation class
# :class:`~gyptis.Grating`, where we specify the
# Transverse Electric polarization case (electric field out of plane
# :math:`\boldsymbol{E} = E_z \boldsymbol{e_z}`) and the ``degree`` of
# Lagrange finite elements.

angle = (90 - theta0) * gy.pi / 180

pw = gy.PlaneWave(lambda0, angle, dim=2)

gratingTE = gy.Grating(geom, epsilon, mu, source=pw, polarization="TM", degree=2)

gratingTE.N_d_order = 1
gratingTE.solve()
effs_TE = gratingTE.diffraction_efficiencies(1, orders=True)
E = gratingTE.solution["total"]

### reference
T_ref = dict(TM=[0.2070, 1.0001], TE=[0.8187, 1.0001])


print("Transmission coefficient")
print(" order      ref       calc")
print("--------------------------------")
print(f"   0       {T_ref['TM'][0]:.4f}    {effs_TE['T'][1]:.4f} ")
print(f"  sum      {T_ref['TM'][1]:.4f}    {effs_TE['B']:.4f}   ")

######################################################################
# We switch to TE polarization

gratingTM = gy.Grating(geom, epsilon, mu, source=pw, polarization="TE", degree=2)
gratingTM.solve()
effs_TM = gratingTM.diffraction_efficiencies(1, orders=True)
H = gratingTM.solution["total"]


######################################################################
# Let's visualize the fields

fig, ax = plt.subplots(1, 2)
ylim = geom.y_position["substrate"], geom.y_position["pml_top"]
gratingTE.plot_field(ax=ax[0])
gratingTE.plot_geometry(ax=ax[0])
ax[0].set_ylim(ylim)
ax[0].set_axis_off()
ax[0].set_title("$E_z$ (TM)")
gratingTM.plot_field(ax=ax[1])
gratingTM.plot_geometry(ax=ax[1])
ax[1].set_ylim(ylim)
ax[1].set_axis_off()
ax[1].set_title("$H_z$ (TE)")
fig.tight_layout()
fig.show()

######################################################################
# Results are in good agreement with the reference

print("Transmission coefficient")
print(" order      ref       calc")
print("--------------------------------")
print(f"   0       {T_ref['TE'][0]:.4f}    {effs_TM['T'][1]:.4f} ")
print(f"  sum      {T_ref['TE'][1]:.4f}    {effs_TM['B']:.4f}   ")
