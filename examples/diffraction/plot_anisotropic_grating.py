# -*- coding: utf-8 -*-
"""
2D Dielectric Grating
=====================

Example of a dielectric diffraction grating.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gyptis import dolfin
from gyptis.grating_2d import *
from gyptis.plotting import *

plt.ion()

##############################################################################
# We will study a classical benchmark of a dielectric grating
# and compare with results given in [PopovGratingBook]_.

fig, ax = plt.subplots(3, 2, figsize=(7, 9))


lambda0 = 633
period = 600

width_bottom, width_top = 500, 300
height = 600

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
        "pml_bottom": 0.7 * pmesh,
        "substrate": pmesh * 2.25 ** 0.5,
        "groove": pmesh,
        "rod": pmesh * 2.59 ** 0.5,
        "superstrate": pmesh,
        "pml_top": 0.7 * pmesh,
    }
)


model = Layered2D(period, thicknesses, kill=False)
groove = model.layers["groove"]
substrate = model.layers["substrate"]
y0 = model.y_position["groove"]
P = []
P.append(model.add_point(-width_bottom / 2, y0, 0))
P.append(model.add_point(width_bottom / 2, y0, 0))
P.append(model.add_point(width_top / 2, y0 + height, 0))
P.append(model.add_point(-width_top / 2, y0 + height, 0))
L = [
    model.add_line(P[0], P[1]),
    model.add_line(P[1], P[2]),
    model.add_line(P[2], P[3]),
    model.add_line(P[3], P[0]),
]
cl = model.add_curve_loop(L)
rod = model.add_plane_surface(model.dimtag(cl, 1)[0])
substrate, groove, rod = model.cut([substrate, groove], rod)
model.add_physical(rod, "rod")
model.add_physical(groove, "groove")
model.add_physical(substrate, "substrate")
mesh_size = {d: lambda0 / param for d, param in mesh_param.items()}
model.set_mesh_size(mesh_size)

model.build(
    interactive=False,
    generate_mesh=True,
    write_mesh=True,
    read_info=True,
)
all_domains = model.subdomains["surfaces"]
domains = [k for k in all_domains.keys() if k not in ["pml_bottom", "pml_top"]]

epsilon = {d: 1 for d in domains}
mu = {d: 1 for d in domains}

epsilon["substrate"] = 2.25
epsilon["rod"] = np.array([[2.592, 0.251, 0], [0.251, 2.592, 0], [0, 0, 2.829]])


grating = Grating2D(
    model,
    epsilon,
    mu,
    lambda0=lambda0,
    degree=2,
)

for jangle, angle in enumerate([0, -20, -40]):
    grating.theta0 = angle * pi / 180
    grating.polarization = "TE"
    grating.N_d_order = 2
    grating.prepare()
    grating.weak_form()
    grating.assemble()
    grating.build_system()
    grating.solve()
    effs_TE = grating.diffraction_efficiencies(orders=True)

    E = grating.solution["total"]
    print(f"angle = {angle}, {grating.polarization} polarization")
    print("--------------------------------")
    print("R: ", effs_TE["R"])
    print("T: ", effs_TE["T"])

    grating.polarization = "TM"
    grating.prepare()
    grating.weak_form()
    grating.assemble()
    grating.build_system()
    grating.solve()
    effs_TM = grating.diffraction_efficiencies(orders=True)

    H = grating.solution["total"]
    print(f"angle = {angle}, {grating.polarization} polarization")
    print("--------------------------------")
    print("R: ", effs_TM["R"])
    print("T: ", effs_TM["T"])

    ylim = model.y_position["substrate"], model.y_position["pml_top"]

    nper = 8
    id = model.subdomains["surfaces"]["rod"]
    d = model.period
    alpha = grating.alpha

    def phase_shift(i, alpha=alpha, d=d, degree=2):
        phasor_re = dolfin.Expression(
            f"cos(i*alpha*d)", i=i, alpha=alpha, d=d, degree=degree
        )
        phasor_im = dolfin.Expression(
            f"sin(i*alpha*d)", i=i, alpha=alpha, d=d, degree=degree
        )
        return Complex(phasor_re, phasor_im)

    vminTE, vmaxTE = -1.5, 1.7
    plt.sca(ax[jangle][0])
    for i in range(nper):
        t = mpl.transforms.Affine2D().translate(i * d, 0)
        f = E * phase_shift(i)
        cbTE = dolfin.plot(
            f.real,
            cmap="RdBu_r",
            transform=t + ax[jangle][0].transData,
            vmin=vminTE,
            vmax=vmaxTE,
        )
        plot_subdomains(grating.markers, domain=id, shift=(i * d, 0))
        cbTE.set_clim(vminTE, vmaxTE)

    plt.ylim(ylim)
    plt.xlim(-d / 2, nper * d - d / 2)
    plt.axis("off")
    ax[jangle][0].axhline(0, 0, 1, color="k")

    vminTM, vmaxTM = -2.5, 2.5
    plt.sca(ax[jangle][1])
    for i in range(nper):
        t = mpl.transforms.Affine2D().translate(i * d, 0)
        f = H * phase_shift(i)
        cbTM = dolfin.plot(
            f.real,
            cmap="RdBu_r",
            transform=t + ax[jangle][1].transData,
            vmin=vminTM,
            vmax=vmaxTM,
        )
        plot_subdomains(grating.markers, domain=id, shift=(i * d, 0))
        cbTM.set_clim(vminTM, vmaxTM)
    plt.ylim(ylim)
    plt.xlim(-d / 2, nper * d - d / 2)
    plt.axis("off")
    ax[jangle][1].axhline(0, 0, 1, color="k")
    ax[jangle][0].set_title(fr"$\theta = {angle}\degree$")
    ax[jangle][1].set_title(fr"$\theta = {angle}\degree$")

    plt.tight_layout()


divider = make_axes_locatable(ax[0, 0])
cax = divider.new_vertical(size="5%", pad=0.6)
fig.add_axes(cax)
mTE = plt.cm.ScalarMappable(cmap="RdBu")
mTE.set_clim(vminTE, vmaxTE)
cbarTE = fig.colorbar(mTE, cax=cax, orientation="horizontal")
cax.set_title(r"${\rm Re}\, E_z$ (TE)", fontsize=20)

divider = make_axes_locatable(ax[0, 1])
cax = divider.new_vertical(size="5%", pad=0.6)
mTM = plt.cm.ScalarMappable(cmap="RdBu")
mTM.set_clim(vminTM, vmaxTM)
fig.add_axes(cax)
cbarTM = fig.colorbar(mTM, cax=cax, orientation="horizontal")
cax.set_title(r"${\rm Re}\, H_z$ (TM)", fontsize=20)

plt.tight_layout()
plt.subplots_adjust(wspace=0.22)


#
######################################################################
#
# .. [PopovGratingBook] T. Antonakakis et al.,
#   Gratings: Theory and Numeric Applications.
#   AMU,(PUP), CNRS, ECM, 2014.
#   `<https://www.fresnel.fr/files/gratings/Second-Edition/>`_
