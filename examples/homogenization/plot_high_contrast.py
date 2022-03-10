#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

"""
High contrast homogenization 
============================

Metamaterial with high index inclusions
"""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy


def analytical_mueff(k, eps_i, Nmax=10):
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


##############################################################################
# Results are compared with :cite:p:`Felbacq2005`.


d = 1
v = (d, 0), (0, d)
a = d / 2
lmin = a / 50
eps_i = 200 - 5j

##############################################################################
# Build the lattice


lattice = gy.Lattice(dim=2, vectors=v)
incl = lattice.add_square(a / 2, a / 2, 0, a)
cell = lattice.cut(lattice.cell, incl)
lattice.add_physical(cell, "background")
lattice.set_size("background", lmin)
lattice.build()

##############################################################################
# Build the inclusion

inclusion = gy.Geometry(dim=2)
incl = inclusion.add_square(a / 2, a / 2, 0, a)
inclusion.add_physical(incl, "inclusion")
bnds = inclusion.get_boundaries("inclusion")
inclusion.add_physical(bnds, "inclusion_bnds", dim=1)
inclusion.set_size("inclusion", lmin)
inclusion.build()


##############################################################################
# Materials

epsilon = dict(inclusion=eps_i, background=1)
mu = dict(inclusion=1, background=1)

##############################################################################
# Homogenization model

hom = gy.models.HighContrastHomogenization2D(lattice, inclusion, epsilon, mu, degree=2)

##############################################################################
# Effective permittivity

eps_eff = hom.get_effective_permittivity(scalar=True)
print(eps_eff[0][0])


##############################################################################
# Effective permeability

neigs = 30
wavevector_target = 0.2
mu_eff = hom.get_effective_permeability(
    0.5, neigs=neigs, wavevector_target=wavevector_target
)
mu_eff_ana = analytical_mueff(0.5, eps_i)
print(mu_eff)
print(mu_eff_ana)


##############################################################################
# Plot eigenvalue spectrum

nms = [n for n in range(1, 6)]
knm = (
    np.array([np.pi / a * (n ** 2 + m ** 2) ** 0.5 for n in nms for m in nms])
    / eps_i ** 0.5
)
qnm = hom.eigs["eigenvalues"]
qnm = qnm[qnm.imag > 0]

plt.figure()
plt.plot(knm.real, knm.imag, "s", alpha=0.8, ms=3, label="analytical")
plt.plot(qnm.real, qnm.imag, "ok", ms=1, label="gyptis")
plt.xlabel(r"Re $k$")
plt.ylabel(r"Im $k$")
plt.legend()
plt.title("eigenvalues")
plt.tight_layout()

##############################################################################
# Plot first mie resonance mode

v = hom.eigs["eigenvectors"]
out = gy.plot(v[0])
fig = plt.gcf()
fig.set_size_inches(3.5, 1.4)
ax = fig.axes[:2]
for axis in ax:
    inclusion.plot_subdomains(c="w", ax=axis)
    axis.set_axis_off()
plt.suptitle("fundamental mode")
plt.tight_layout()
plt.show()

##############################################################################
# Plot frequency dispersion of the effective permeability

lambdas = np.linspace(4, 15, 1000) * d
k = 2 * np.pi / lambdas
mu_eff = hom.get_effective_permeability(
    k, neigs=neigs, wavevector_target=wavevector_target
)
mu = analytical_mueff(k, eps_i)

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
plt.suptitle(r"$\varepsilon_{\rm eff} = $" + f"{eps_eff[0][0].real:.3f}")
plt.tight_layout()

##############################################################################
# Compare models


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
    epsilon["groove"] = eps_eff[0][0]
    mu = {d: 1 for d in domains_hom}
    mu["groove"] = analytical_mueff(k, eps_i)
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
