#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

"""
Quasinormal modes and modal expansion of a nanorod
==================================================

Spectral problem for a triangular rod made of a non-dispersive dielectric in vacuum. 
Reconstruction using a quasinormlal mode expansion applied to the scattering by 
a plane wave and the computaion of the local density of states.

"""


# sphinx_gallery_thumbnail_number = -1

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy
import gyptis.utils.data_download as dd
from gyptis import c, pi
from gyptis.utils.helpers import array2function, function2array, project_iterative

##############################################################################
# Reference results are taken from :cite:p:`Vial2014`.

neig = 151
lbox = 6.5
pml_width = 15
pmesh = 6
lmin = 3 / pmesh
eps_rod = 13 - 0.2j


##############################################################################
# Build and mesh the geometry:


geom = gy.BoxPML(
    dim=2,
    box_size=(lbox, lbox),
    pml_width=(pml_width, pml_width),
)

A = geom.add_point(-1, 3, 0)
B = geom.add_point(-1, -2, 0)
C = geom.add_point(3, -1, 0)

AB = geom.add_line(A, B)
BC = geom.add_line(B, C)
CA = geom.add_line(C, A)

loop = geom.add_curve_loop([AB, BC, CA])
tri = geom.add_plane_surface([loop])
tri, box = geom.fragment(tri, geom.box)
geom.add_physical(box, "box")
geom.add_physical(tri, "rod")
geom.set_size("box", lmin)
geom.set_size("rod", lmin / (eps_rod.real ** 0.5))
[geom.set_size(pml, lmin) for pml in geom.pmls]

geom.build()

epsilon = dict(box=1, rod=eps_rod)
mu = dict(box=1, rod=1)


##############################################################################
# Eigenvalue problem:

s_modal = gy.Scattering(
    geom,
    epsilon,
    mu,
    polarization="TM",
    modal=True,
)

s_modal.eigensolve(neig, wavevector_target=0.4)
ev = s_modal.solution["eigenvalues"]
modes = s_modal.solution["eigenvectors"]


##############################################################################
# Plot the spectrum:

a = 1
ev_norma = ev * a / (2 * pi)

eigenvalues_file = dd.download_example_data(
    data_file_name="qnms_triangle.csv",
    example_dir="modal",
)

benchmark_qnms = np.loadtxt(eigenvalues_file, delimiter=",")
omega_ref = (benchmark_qnms[:, 0] - 1j * benchmark_qnms[:, 1]) * 1e14
ev_norma_ref = omega_ref * a * 1e-6 / (2 * pi * c)

fig, axcp = plt.subplots()
plt.plot(ev_norma_ref.real, ev_norma_ref.imag, "ok", label="ref.")
plt.plot(ev_norma.real, ev_norma.imag, "+r", label="gyptis")
plt.xlabel(rf"Re $\tilde{{\omega}}$")
plt.xlabel(rf"Im $\tilde{{\omega}}$")
plt.legend()
plt.tight_layout()

form = s_modal.formulation
form.source_domains = ["rod"]
xi = form.xi.as_subdomain()
chi = form.chi.as_subdomain()
xi_a = form.xi.build_annex(
    domains=form.source_domains, reference=form.reference
).as_subdomain()
chi_a = form.chi.build_annex(
    domains=form.source_domains, reference=form.reference
).as_subdomain()


omega_1 = (1.77 + 0.0636j) * 1e14
omega_2 = (1.90 + 0.101j) * 1e14
ev_norma_ref_1 = omega_1 * a * 1e-6 / (2 * pi * c)
ev_norma_ref_2 = omega_2 * a * 1e-6 / (2 * pi * c)
i1 = np.argmin(abs(ev_norma - ev_norma_ref_1))
i2 = np.argmin(abs(ev_norma - ev_norma_ref_2))

# plt.plot(ev_norma_ref_1.real, ev_norma_ref_1.imag, "xb")
# plt.plot(ev_norma_ref_2.real, ev_norma_ref_2.imag, "xb")
# plt.plot(ev_norma_1.real, ev_norma_1.imag, "sc")
# plt.plot(ev_norma_2.real, ev_norma_2.imag, "sc")


##############################################################################
# Plot modes:

for i, mode_index in enumerate([i1, i2]):
    v = modes[mode_index]
    ## normalize
    Kn = gy.assemble(gy.dot(chi * v, v) * form.dx)
    qnm = v / Kn ** 0.5

    plt.figure()
    gy.plot(qnm.real, cmap="RdBu_r", ax=plt.gca())
    geom.plot_subdomains()
    plt.axis("off")
    plt.title(rf"mode {i+1}: $\tilde{{\omega}}=$ {ev_norma[mode_index]:.4f}")
    plt.pause(0.1)


##############################################################################
# Calculate coupling coefficients:


def get_coupling_coeff(scatt, mode_index, pw, Kn=None):
    ev = scatt.solution["eigenvalues"]
    modes = scatt.solution["eigenvectors"]
    vn = modes[mode_index]
    kn = ev[mode_index]
    k = pw.wavenumber
    if Kn is None:
        Kn = gy.assemble(gy.dot(chi * vn, vn) * form.dx)
    source = form.maxwell(pw.expression, vn, xi - xi_a, chi - chi_a, domain="rod")
    ss = -source[0] + gy.Constant(k) ** 2 * source[1]
    Jn = gy.assemble(ss) / Kn
    Pn = Jn / (k ** 2 - kn ** 2)
    return Pn.real + 1j * Pn.imag


wls = np.linspace(9, 11, 21)
angles = np.linspace(0, 2 * pi, 21)

fig, ax = plt.subplots(2, 1, figsize=(5, 4))

for i, mode_index in enumerate([i1, i2]):
    vn = modes[mode_index]
    Kn = gy.assemble(gy.dot(chi * vn, vn) * form.dx)
    coupling = []
    for angle in angles:
        # print(f"θ = {angle*180/pi}°")
        coupling_ = []
        for wavelength in wls:
            # print(f"λ = {wavelength}μm")
            pw = gy.PlaneWave(
                wavelength=wavelength,
                angle=pi / 2 + angle,
                dim=2,
                domain=geom.mesh,
                degree=2,
            )
            Pn = get_coupling_coeff(s_modal, mode_index, pw, Kn=Kn)
            coupling_.append(Pn)
        coupling.append(coupling_)
    coupling = np.array(coupling)

    q = ax[i].contourf(
        angles * 180 / pi, wls, np.abs(coupling).T, cmap="magma", levels=101
    )
    plt.colorbar(q, ax=ax[i])
    plt.pause(0.1)

ax[0].annotate("mode 1", xy=(0.01, 0.05), xycoords="axes fraction", c="w")
ax[1].annotate("mode 2", xy=(0.01, 0.05), xycoords="axes fraction", c="w")
ax[0].set_xticklabels("")
ax[1].set_xlabel(r"$\theta$ (degree)")
ax[0].set_ylabel(r"$\lambda$ ($\mu m$)")
ax[1].set_ylabel(r"$\lambda$ ($\mu m$)")
plt.suptitle("coupling coefficients")
plt.tight_layout()


##############################################################################
# Comparison, first we compute the scattering from a plane wave.

wavelength = 10.2
angle = 143 * pi / 180
pw = gy.PlaneWave(
    wavelength=wavelength, angle=pi / 2 + angle, dim=2, domain=geom.mesh, degree=2
)
s_direct = gy.Scattering(
    geom,
    epsilon,
    mu,
    source=pw,
    polarization="TM",
)
s_direct.solve()
sf = s_direct.solution["diffracted"]

##############################################################################
# Reconstruction:

Nmodes = 50
mode_indexes = range(neig)

PNS = []
for mode_index in mode_indexes:
    PNS.append(get_coupling_coeff(s_modal, mode_index, pw))
mode_indexes_rec = np.argsort(np.abs(PNS))
mode_indexes_rec = np.flipud(mode_indexes_rec)[:Nmodes]
ev_qmem = ev_norma[mode_indexes_rec]
# axcp.plot(ev_qmem.real, ev_qmem.imag, "xg")

reconstr = 0
for mode_index in mode_indexes_rec:
    reconstr += gy.Constant(PNS[mode_index]) * modes[mode_index]


##############################################################################
# Vizualize the total field for the direct problem:

fig, ax = plt.subplots(1, 2, figsize=(4.5, 2))
gy.plotcplx(sf, cmap="RdBu_r", ax=ax)
[geom.plot_subdomains(ax=a) for a in ax]
for a in ax:
    geom.plot_subdomains(ax=a)
    a.set_axis_off()
ax[0].set_title(r"Re $E_z$")
ax[1].set_title(r"Im $E_z$")
plt.suptitle("direct")
plt.axis("off")
plt.tight_layout()
plt.pause(0.1)

##############################################################################
# Vizualize the total field for the quasimodal expansion:

fig, ax = plt.subplots(1, 2, figsize=(4.5, 2))
gy.plotcplx(reconstr, cmap="RdBu_r", ax=ax)
[geom.plot_subdomains(ax=a) for a in ax]
for a in ax:
    geom.plot_subdomains(ax=a)
    a.set_axis_off()
ax[0].set_title(r"Re $E_z$")
ax[1].set_title(r"Im $E_z$")
plt.suptitle("QMEM")
plt.axis("off")
plt.tight_layout()
plt.pause(0.1)


##############################################################################
# Compute the LDOS

ls = gy.LineSource(wavelength=10.2, position=(0, 0), domain=geom.mesh, degree=2)
s_direct = gy.Scattering(
    geom,
    epsilon,
    mu,
    source=ls,
    polarization="TM",
    degree=2,
)

# LDOS in vaccum
ldos_vac = 2 * s_direct.source.pulsation / (pi * c ** 2) * 0.25

nx, ny = 21, 21
X = np.linspace(-lbox / 2, lbox / 2, nx)
Y = np.linspace(-lbox / 2, lbox / 2, ny)
ldos = np.zeros((nx, ny))

for j, y in enumerate(Y):
    for i, x in enumerate(X):
        # print(f"{i}, {j}")
        ldos[i, j] = s_direct.local_density_of_states(x, y)


plt.figure()
plt.contourf(X, Y, ldos.T / ldos_vac, cmap="Spectral_r", levels=101)
geom.plot_subdomains()
plt.axis("scaled")
plt.xlim(-lbox / 2, lbox / 2)
plt.ylim(-lbox / 2, lbox / 2)
plt.colorbar()
plt.suptitle("normalized LDOS (direct)")
plt.tight_layout()
plt.pause(0.1)


##############################################################################
# Compute the LDOS with the quasinormal mode expansion:

k = ls.wavenumber
mode_indexes_rec = range(neig)

ldos_qmem_array = 0
for mode_index in mode_indexes_rec:
    vn = modes[mode_index]
    Kn = gy.assemble(gy.dot(chi * vn, vn) * form.dx)
    vn = project_iterative(vn, s_modal.formulation.real_function_space)
    vn = function2array(vn.real) + 1j * function2array(vn.imag)
    kn = ev[mode_index]
    ldos_qmem_array += ((vn * vn) / (Kn * (k ** 2 - kn ** 2))).imag

ldos_qmem = array2function(ldos_qmem_array, s_modal.formulation.real_function_space)
ldos_qmem *= -2 * ls.pulsation / (pi * c ** 2)
Z = -ldos_qmem / ldos_vac
Za = 2 * ls.pulsation / (pi * c ** 2) * ldos_qmem_array / ldos_vac

mini = 0.325
maxi = 2.8  # Za.max()
nlev = 101

plt.figure()
p, cb = gy.plot(Z, cmap="Spectral_r", levels=nlev, ax=plt.gca(), vmin=mini, vmax=maxi)
cb.remove()
geom.plot_subdomains()
plt.axis("scaled")
plt.xlim(-lbox / 2, lbox / 2)
plt.ylim(-lbox / 2, lbox / 2)
plt.suptitle("normalized LDOS (QMEM)")
plt.tight_layout()
plt.pause(0.1)
m = plt.cm.ScalarMappable(cmap="Spectral_r")
m.set_array(Za)
m.set_clim(mini, maxi)
_ = plt.colorbar(m, boundaries=np.linspace(mini, maxi, nlev))
