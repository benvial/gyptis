# -*- coding: utf-8 -*-
"""
2D Anisotropic Grating
=======================

Example of diffraction grating with trapezoidal ridges made from an anisotropic material.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from collections import OrderedDict
from gyptis import Layered, Grating, PlaneWave
from gyptis.plot import *

##############################################################################
# We will study this benchmark and compare with results
# given in [PopovGratingBook]_.

fig, ax = plt.subplots(3, 2, figsize=(7, 9))


lambda0 = 633
period = 600

width_bottom, width_top = 500, 300
height = 600

pmesh = 6

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
        "pml_bottom": 0.7 * pmesh,
        "substrate": pmesh * 2.25 ** 0.5,
        "groove": pmesh,
        "rod": pmesh * 2.59 ** 0.5,
        "superstrate": pmesh,
        "pml_top": 0.7 * pmesh,
    }
)


geom = Layered(2, period, thicknesses)
groove = geom.layers["groove"]
substrate = geom.layers["substrate"]
y0 = geom.y_position["groove"]
P = []
P.append(geom.add_point(-width_bottom / 2, y0, 0))
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
substrate, groove, rod = geom.cut([substrate, groove], rod)
geom.add_physical(rod, "rod")
geom.add_physical(groove, "groove")
geom.add_physical(substrate, "substrate")
mesh_size = {d: lambda0 / param for d, param in mesh_param.items()}
geom.set_mesh_size(mesh_size)

geom.build(
    interactive=False, generate_mesh=True, write_mesh=True, read_info=True,
)
all_domains = geom.subdomains["surfaces"]
domains = [k for k in all_domains.keys() if k not in ["pml_bottom", "pml_top"]]

epsilon = {d: 1 for d in domains}
mu = {d: 1 for d in domains}

epsilon["substrate"] = 2.25
epsilon["rod"] = np.array([[2.592, 0.251, 0], [0.251, 2.592, 0], [0, 0, 2.829]])



for jangle, angle in enumerate([0, 20, 40]):
    
    angle_degree = (90 - angle) * np.pi / 180
    
    pw = PlaneWave(lambda0, angle_degree, dim=2)
    grating_TE = Grating(geom, epsilon, mu, source=pw, polarization="TE", degree=2)
    grating_TE.solve()
    effs_TE = grating_TE.diffraction_efficiencies(2, orders=True)

    E = grating_TE.solution["total"]
    print(f"angle = {angle}, TE polarization")
    print("--------------------------------")
    print("R: ", effs_TE["R"])
    print("T: ", effs_TE["T"])

    ylim = geom.y_position["substrate"], geom.y_position["pml_top"]
    d = grating_TE.period
    nper = 8

    vminTE, vmaxTE = -1.5, 1.7
    plt.sca(ax[jangle][0])
    per_plots, cb = grating_TE.plot_field(nper=nper)
    cb.remove()
    scatt_lines, layers_lines = grating_TE.plot_geometry(nper=nper, c="k")
    [layers_lines[i].remove() for i in [0, 1, 3, 4]]
    plt.ylim(ylim)
    plt.xlim(-d / 2, nper * d - d / 2)
    plt.axis("off")
    
    #### TM
    
    grating_TM = Grating(geom, epsilon, mu, source=pw, polarization="TM", degree=2)

    grating_TM.solve()
    effs_TM = grating_TM.diffraction_efficiencies(2, orders=True)

    H = grating_TM.solution["total"]
    print(f"angle = {angle}, TM polarization")
    print("--------------------------------")
    print("R: ", effs_TM["R"])
    print("T: ", effs_TM["T"])

    vminTM, vmaxTM = -2.5, 2.5
    plt.sca(ax[jangle][1])
    per_plots, cb = grating_TM.plot_field(nper=nper)
    cb.remove()
    scatt_lines, layers_lines = grating_TM.plot_geometry(nper=nper, c="k")
    [layers_lines[i].remove() for i in [0, 1, 3, 4]]
    plt.ylim(ylim)
    plt.xlim(-d / 2, nper * d - d / 2)
    plt.axis("off")

    ax[jangle][0].set_title(fr"$\theta = {angle}\degree$")
    ax[jangle][1].set_title(fr"$\theta = {angle}\degree$")

    # plt.tight_layout()


divider = make_axes_locatable(ax[0, 0])
cax = divider.new_vertical(size="5%", pad=0.5)
fig.add_axes(cax)
mTE = plt.cm.ScalarMappable(cmap="RdBu")
mTE.set_clim(vminTE, vmaxTE)
cbarTE = fig.colorbar(mTE, cax=cax, orientation="horizontal")
cax.set_title(r"${\rm Re}\, E_z$ (TE)", fontsize=20)

divider = make_axes_locatable(ax[0, 1])
cax = divider.new_vertical(size="5%", pad=0.5)
mTM = plt.cm.ScalarMappable(cmap="RdBu")
mTM.set_clim(vminTM, vmaxTM)
fig.add_axes(cax)
cbarTM = fig.colorbar(mTM, cax=cax, orientation="horizontal")
cax.set_title(r"${\rm Re}\, H_z$ (TM)", fontsize=20)

plt.tight_layout()
plt.subplots_adjust(wspace=-0.1)


#
######################################################################
#
# .. [PopovGratingBook] T. Antonakakis et al.,
#   Gratings: Theory and Numeric Applications.
#   AMU,(PUP), CNRS, ECM, 2014.
#   `<https://www.fresnel.fr/files/gratings/Second-Edition/>`_
