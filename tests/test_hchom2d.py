#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

import matplotlib.pyplot as plt

plt.ion()
plt.close("all")
from collections import OrderedDict

import numpy as np
import pytest

import gyptis as gy

d = 1
v = (d, 0), (0, d)
a = d / 2
lmin = a / 10

lattice = gy.Lattice(dim=2, vectors=v)
incl = lattice.add_square(a / 2, a / 2, 0, a)
cell = lattice.cut(lattice.cell, incl)
lattice.add_physical(cell, "background")
lattice.set_size("background", lmin)
lattice.build()

inclusion = gy.Geometry(dim=2)
incl = inclusion.add_square(a / 2, a / 2, 0, a)
inclusion.add_physical(incl, "inclusion")
bnds = inclusion.get_boundaries(incl)
inclusion.add_physical(bnds, "inclusion_bnds", dim=1)
inclusion.set_size("inclusion", lmin)
inclusion.build()

eps_i = 200 - 5j
epsilon = dict(inclusion=eps_i, background=1)
mu = dict(inclusion=1, background=1)


def analytical_mueff(k, Nmax=10):
    nms = [2 * n + 1 for n in range(Nmax)]
    mu = 1
    for n in nms:
        for m in nms:
            knm = np.pi / a * (n ** 2 + m ** 2) ** 0.5
            qn = np.pi / a * n
            pm = np.pi / a * m
            alpha = 2 / qn * 2 / pm
            norm = (a / 2) ** 2
            mu += -(k ** 2 * eps_i) / (k ** 2 * eps_i - knm ** 2) * alpha ** 2 / norm
    return mu


hom = gy.models.HighContrastHomogenization2D(lattice, inclusion, epsilon, mu, degree=2)
eps_eff = hom.get_effective_permittivity(scalar=True)
print(eps_eff)

mu_eff = hom.get_effective_permeability(0.5)
print(mu_eff)
mu_eff_ana = analytical_mueff(0.5)
print(mu_eff_ana)

assert np.allclose(mu_eff.tocomplex(), mu_eff_ana, 1e-3)

lambdas = np.linspace(4, 15, 1000) * d
k = 2 * np.pi / lambdas

mu_eff = hom.get_effective_permeability(k)
mu = analytical_mueff(k)

# Qs = np.array(Qs) / eps_i ** 0.5
# # Es /= eps_i ** 0.5
# plt.figure()
# plt.plot(Qs.real, Qs.imag, "o")
# plt.plot(Es.real, Es.imag, "+")
# # plt.ylim(-1,1)


# plt.figure()
fig, ax = plt.subplots(2, 1, figsize=(2.7, 3.5))
ax[0].plot(lambdas / d, mu.real, "-", label="analytical", lw=2, alpha=0.25, c="#44304b")
ax[1].plot(
    lambdas / d, mu.imag, "--", label="analytical", lw=2, alpha=0.25, c="#44304b"
)
ax[0].plot(lambdas / d, mu_eff.real, "-", label="gyptis", c="#d78130")
ax[1].plot(lambdas / d, mu_eff.imag, "-", label="gyptis", c="#d78130")
# plt.plot(Qs.real, 0*Qs.imag,"or")
# plt.plot(Es.real, 0*Es.imag,"+b")
ax[0].set_ylabel(r"Re $\mu_{\rm eff}$")
ax[1].set_ylabel(r"Im $\mu_{\rm eff}$")
ax[0].set_xlabel(r"$\lambda/d$")
ax[1].set_xlabel(r"$\lambda/d$")
ax[0].legend()
ax[1].legend()
plt.tight_layout()

####### grating


WLS = np.linspace(4, 15, 101) * d
wavelength_max = WLS.max()
wavelength_min = WLS.min()

thicknesses = OrderedDict(
    {
        "pml_bottom": wavelength_max,
        "substrate": wavelength_max,
        "groove": 3 * d,
        "superstrate": wavelength_max,
        "pml_top": wavelength_max,
    }
)

n_rod = eps_i.real ** 0.5
pmesh = 3
pmesh_rod = pmesh * 1
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


geom = gy.Layered(2, d, thicknesses)
groove = geom.layers["groove"]
y0 = geom.y_position["groove"] + thicknesses["groove"] / 2
rod = [geom.add_square(-a / 2, (d - a) / 2 + i * d, 0, a) for i in range(3)]
# rod, groove,substrate ,superstrate= geom.fragment(groove, rod)
*rod, groove = geom.fragment(groove, rod)
geom.add_physical(rod, "rod")
geom.add_physical(groove, "groove")
# geom.add_physical(substrate, "substrate")
# geom.add_physical(superstrate, "superstrate")
mesh_size = {d: wavelength_min / param for d, param in mesh_param.items()}
geom.set_mesh_size(mesh_size)
geom.build()
domains = geom.subdomains["surfaces"]


geom_hom = gy.Layered(2, d / 10, thicknesses)
mesh_size = {d: wavelength_min / param for d, param in mesh_param.items() if d != "rod"}
geom_hom.set_mesh_size(mesh_size)
geom_hom.build()
domains_hom = geom_hom.subdomains["surfaces"]


angle = 0 * gy.pi / 180
trans = []
trans_eff = []
for wavelength in WLS:
    k = 2 * gy.pi / wavelength
    print(f">> wavelength = {wavelength}")
    pw = gy.PlaneWave(wavelength, angle, dim=2)

    epsilon = {d: 1 for d in domains}
    epsilon["rod"] = eps_i
    mu = {d: 1 for d in domains}

    grat = gy.Grating(geom, epsilon, mu, source=pw, polarization="TE", degree=2)
    grat.solve()
    effs = grat.diffraction_efficiencies(cplx_effs=True)
    t = effs["T"][0]

    print("t grating = ", t)
    trans.append(t.tocomplex())

    epsilon = {d: 1 for d in domains_hom}
    epsilon["groove"] = eps_eff[0][0]  # eps_eff
    mu = {d: 1 for d in domains_hom}
    mu["groove"] = analytical_mueff(k)
    grat = gy.Grating(geom_hom, epsilon, mu, source=pw, polarization="TE", degree=2)
    grat.solve()
    effs = grat.diffraction_efficiencies(cplx_effs=True)
    t = effs["T"][0]
    print("t homogenized = ", t)
    trans_eff.append(t.tocomplex())

trans = np.array(trans)
trans_eff = np.array(trans_eff)
plt.figure()

plt.clf()
plt.plot(WLS / d, trans.real, "-", c="#2c2eb1", label="Re grating")
plt.plot(WLS / d, trans.imag, "-", c="#2cb17d", label="Im grating")
plt.plot(WLS / d, trans_eff.real, "--", c="#2c2eb1", label="Re effective")
plt.plot(WLS / d, trans_eff.imag, "--", c="#2cb17d", label="Im effective")
plt.legend()
plt.ylabel(r"Transmission")
plt.xlabel(r"$\lambda/d$")
plt.tight_layout()

#
#
# ######### slab
#
# wavelength = 9.65 * d
# pmesh = 3
#
# Nx = 40
# Ny = 5
# lbox_x = Nx * d + 1 * wavelength
# lbox_y = Ny * d + 4 * wavelength
#
#
# lmin = wavelength / pmesh
# geom = gy.BoxPML(
#     dim=2,
#     box_size=(lbox_x, lbox_y),
#     box_center=(0, 0),
#     pml_width=(wavelength, wavelength),
# )
# box = geom.box
# rods = [
#     geom.add_square((-Nx / 2 + i) * d, (-Ny / 2 + j) * d, 0, a)
#     for i in range(Nx)
#     for j in range(Ny)
# ]
# *rods, box = geom.fragment(box, rods)
# geom.add_physical(box, "box")
# geom.add_physical(rods, "rods")
# [geom.set_size(pml, lmin * 0.7) for pml in geom.pmls]
# geom.set_size("box", lmin)
# geom.set_size("rods", lmin/eps_i.real**0.5)
# geom.build()
#
#
# pw = gy.GaussianBeam(
#     wavelength=wavelength,
#     angle=3*gy.pi/4,
#     waist=wavelength,
#     position=(0, 0),
#     dim=2,
#     domain=geom.mesh,
#     degree=2,
# )
#
# epsilon = dict(box=1, rods=eps_i)
# mu = dict(box=1, rods=1)
# s = gy.Scattering(
#     geom,
#     epsilon,
#     mu,
#     pw,
#     degree=2,
#     polarization="TE",
# )
#
# s.solve()
# plt.figure(figsize=(4,1.8))
# s.plot_field()
# ax = plt.gca()
# # for i in range(N):
# #     cir = plt.Circle((-N / 2 * d + i * d, 0), R, lw=0.3, color="w", fill=False)
# #     ax.add_patch(cir)
# plt.xlabel("x (nm)")
# plt.ylabel("y (nm)")
# plt.title(title)
# plt.tight_layout()
