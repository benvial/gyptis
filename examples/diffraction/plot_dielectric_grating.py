# -*- coding: utf-8 -*-
"""
2D Dielectric Grating
=====================

Example of a dielectric diffration grating.
"""

import dolfin as df

from gyptis.grating_2d import *
from gyptis.plotting import *

##############################################################################
# We will study a classical benchmark of a dielectric grating
# and compare with results given in [PopovGratingBook]_.


lambda0 = 600
theta0 = -20 * pi / 180

period = 800
h = 400
w = 600

pmesh = 18

thicknesses = OrderedDict(
    {
        "pml_bottom": 1 * lambda0,
        "substrate": 1 * lambda0,
        "groove": 2 * h * 1.5,
        "superstrate": 1 * lambda0,
        "pml_top": 1 * lambda0,
    }
)

mesh_param = dict(
    {
        "pml_bottom": 0.7 * pmesh,
        "substrate": pmesh,
        "groove": pmesh,
        "rod": pmesh * 1.4,
        "superstrate": pmesh,
        "pml_top": 0.7 * pmesh,
    }
)

######################################################################
# Let's create the geometry

model = Layered2D(period, thicknesses, kill=False)
groove = model.layers["groove"]
y0 = model.y_position["groove"] + thicknesses["groove"] / 2
rod = model.addEllipse(0, y0, 0, w / 2, h / 2)
gmsh.model.occ.addCurveLoop([rod], rod)
rod = model.addPlaneSurface([rod])
rod, groove = model.fragmentize(groove, rod, removeTool=True)
model.add_physical(rod, "rod")
model.add_physical(groove, "groove")

mesh_size = {d: lambda0 / param for d, param in mesh_param.items()}
model.set_mesh_size(mesh_size)

model.build(
    interactive=False,
    generate_mesh=True,
    write_mesh=True,
    read_info=True,
)

domains = [k for k in thicknesses.keys() if k not in ["pml_bottom", "pml_top"]]

epsilon = {d: 1 for d in domains}
mu = {d: 1 for d in domains}

epsilon["rod"] = 1.4 ** 2
mu["rod"] = 1
polarization = "TE"

grating = Grating2D(
    model,
    epsilon,
    mu,
    polarization=polarization,
    lambda0=lambda0,
    theta0=theta0,
    degree=2,
)

grating.N_d_order = 1

grating.weak_form()
grating.assemble()
grating.solve(direct=True)
effs_TE = grating.diffraction_efficiencies(orders=True)

E = grating.u + grating.ustack_coeff

### reference
T_ref = dict(TE=[0.2070, 1.0001], TM=[0.8187, 1.0001])


print("Transmission coefficient")
print(" order      ref       calc")
print("--------------------------------")
print(f"   0       {T_ref['TE'][0]:.4f}    {effs_TE['T'][1]:.4f} ")
print(f"  sum      {T_ref['TE'][1]:.4f}    {effs_TE['B']:.4f}   ")


######################################################################
# We switch to TM polarization

grating.polarization = "TM"
grating.weak_form()
grating.assemble()
grating.solve()
effs_TM = grating.diffraction_efficiencies(orders=True)

H = grating.u + grating.ustack_coeff

import matplotlib.pyplot as plt

plt.ion()

ylim = model.y_position["substrate"], model.y_position["pml_top"]

fig, ax = plt.subplots(1, 2)
plt.sca(ax[0])
cb = df.plot(E.real, cmap="RdBu_r")
plot_subdomains(grating.markers)
plt.ylim(ylim)
plt.colorbar(cb)
plt.axis("off")
ax[0].set_title("$E_z$ (TE)")
plt.tight_layout()

plt.sca(ax[1])
cb = df.plot(H.real, cmap="RdBu_r")
plot_subdomains(grating.markers)
plt.ylim(ylim)
plt.colorbar(cb)
plt.axis("off")
ax[1].set_title("$H_z$ (TM)")
plt.tight_layout()

print("Transmission coefficient")
print(" order      ref       calc")
print("--------------------------------")
print(f"   0       {T_ref['TM'][0]:.4f}    {effs_TM['T'][1]:.4f} ")
print(f"  sum      {T_ref['TM'][1]:.4f}    {effs_TM['B']:.4f}   ")

#
# print(f"{R_calc['TE']}")

######################################################################
#
# .. [PopovGratingBook] T. Antonakakis et al.,
#   Gratings: Theory and Numeric Applications.
#   AMU,(PUP), CNRS, ECM, 2014.
#   `<https://www.fresnel.fr/files/gratings/Second-Edition/>`_
