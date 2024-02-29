#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
Core shell nanorod
==================

Scattering by a dielectric cylinder coated with silver.
"""


# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy
import gyptis.utils.data_download as dd
from gyptis import c, pi
from gyptis.utils import adaptive_sampler

##############################################################################
# Reference results are taken from :cite:p:`Jandieri2015`.
# We first define a function for the Drude Lorentz model of silver permittivity.


def epsilon_silver(omega):
    eps_inf = 3.91
    omega_D = 13420e12
    gamma_D = 84e12
    Omega_L = 6870e12
    Gamma_L = 12340e12
    delta_eps = 0.76
    epsilon = (
        eps_inf
        - omega_D**2 / (omega * (omega + 1j * gamma_D))
        - delta_eps
        * Omega_L**2
        / ((omega**2 - Omega_L**2) + 1j * Gamma_L * omega)
    )
    return np.conj(epsilon)


wavelength_min = 250
wavelength_max = 800
wl = np.linspace(wavelength_min, wavelength_max, 100)
omega = 2 * pi * c / (wl * 1e-9)
epsAg = epsilon_silver(omega)
fig, ax = plt.subplots(1, 2)
ax[0].plot(wl, epsAg.real, c="#7b6eaf")
ax[1].plot(wl, epsAg.imag, c="#c63c71")
ax[0].set_xlabel("wavelength (nm)")
ax[1].set_xlabel("wavelength (nm)")
ax[0].set_ylabel(r"Re $\varepsilon$")
ax[1].set_ylabel(r"Im $\varepsilon$")
plt.suptitle("silver permittivity")
plt.tight_layout()

##############################################################################
# Now we create the geometry and mesh

pmesh = 20
degree = 2
wavelength = 452
eps_core = 2


def create_geometry(wavelength, pml_width):
    R1 = 60
    R2 = 30
    Rcalc = 2 * R1
    lmin = wavelength / pmesh
    omega = 2 * pi * c / (wavelength * 1e-9)
    epsAg = epsilon_silver(omega)

    nAg = abs(epsAg.real) ** 0.5
    ncore = abs(eps_core.real) ** 0.5

    lbox = Rcalc * 2 * 1.1
    geom = gy.BoxPML(
        dim=2,
        box_size=(lbox, lbox),
        pml_width=(pml_width, pml_width),
        Rcalc=Rcalc,
    )
    box = geom.box
    shell = geom.add_circle(0, 0, 0, R1)
    out = geom.fragment(shell, box)
    box = out[1:3]
    shell = out[0]
    core = geom.add_circle(0, 0, 0, R2)
    core, shell = geom.fragment(core, shell)
    geom.add_physical(box, "box")
    geom.add_physical(core, "core")
    geom.add_physical(shell, "shell")
    [geom.set_size(pml, lmin * 1) for pml in geom.pmls]
    geom.set_size("box", lmin)
    geom.set_size("core", 0.5 * lmin / ncore)
    geom.set_size("shell", 0.5 * lmin / nAg)
    geom.build()
    return geom


geom = create_geometry(wavelength, pml_width=wavelength)


##############################################################################
# Define the incident plane wave and materials

pw = gy.PlaneWave(
    wavelength=wavelength, angle=0, dim=2, domain=geom.mesh, degree=degree
)
omega = 2 * pi * c / (wavelength * 1e-9)
epsilon = dict(box=1, core=eps_core, shell=epsilon_silver(omega))


##############################################################################
# Scattering problem

s = gy.Scattering(
    geom,
    epsilon,
    source=pw,
    degree=degree,
    polarization="TE",
)
s.solve()
s.plot_field()
geom_lines = geom.plot_subdomains()
plt.xlabel(r"$x$ (nm)")
plt.ylabel(r"$y$ (nm)")
plt.title(r"Re $H_z$")
plt.tight_layout()

##############################################################################
# Compute cross sections and check energy conservation (optical theorem)

cs = s.get_cross_sections()
print("cross sections")
print("--------------")
for k, v in cs.items():
    print(f"{k} {v:.3f}nm")
print("energy balance", (cs["scattering"] + cs["absorption"]) / cs["extinction"])
assert np.allclose(cs["extinction"], cs["scattering"] + cs["absorption"], rtol=1e-12)


##############################################################################
# Compute spectra using adaptive sampling. The function must return a scalar
# (which will be monitored by the sampler) as its first output.


@adaptive_sampler(max_bend=10, max_z_rel=0.001, max_df=0.02)
def cs_vs_wl(wavelength):
    geom = create_geometry(wavelength, pml_width=wavelength)
    pw = gy.PlaneWave(wavelength=wavelength, angle=0, dim=2, domain=geom.mesh, degree=2)
    omega = 2 * pi * c / (wavelength * 1e-9)
    epsilon = dict(box=1, core=2, shell=epsilon_silver(omega))
    s = gy.Scattering(
        geom,
        epsilon,
        source=pw,
        degree=2,
        polarization="TE",
    )
    s.solve()
    cs = s.get_cross_sections()
    return cs["absorption"], cs


wl = np.linspace(wavelength_min, wavelength_max, 50)
wla, out = cs_vs_wl(wl)
cs = [_[1] for _ in out]
scs = np.array([d["scattering"] for d in cs])
acs = np.array([d["absorption"] for d in cs])
ecs = np.array([d["extinction"] for d in cs])
balance = (scs + acs) / ecs

##############################################################################
# Plot energy balance


plt.figure()
plt.plot(
    wla,
    balance,
    c="#df6482",
)
plt.ylabel("energy balance")
plt.xlabel("wavelength (nm)")
plt.xlim(wavelength_min, wavelength_max)
plt.tight_layout()
plt.show()

##############################################################################
# Plot and comparison with benchmark

scs_file = dd.download_example_data(
    data_file_name="scs_r2_30nm.csv",
    example_dir="scattering",
)

acs_file = dd.download_example_data(
    data_file_name="acs_r2_30nm.csv",
    example_dir="scattering",
)

benchmark_scs = np.loadtxt(scs_file, delimiter=",")
benchmark_acs = np.loadtxt(acs_file, delimiter=",")


plt.figure()
plt.plot(
    benchmark_scs[:, 0],
    benchmark_scs[:, 1],
    "-",
    alpha=0.5,
    lw=2,
    c="#df6482",
    label="scattering reference",
)
plt.plot(wla, scs, c="#df6482", label="scattering gyptis")
plt.plot(
    benchmark_acs[:, 0],
    benchmark_acs[:, 1],
    "-",
    alpha=0.5,
    lw=2,
    c="#6e8cd0",
    label="absorption reference",
)
plt.plot(wla, acs, c="#6e8cd0", label="absorption gyptis")
plt.xlabel("wavelength (nm)")
plt.ylabel("cross sections (nm)")
plt.legend()
plt.xlim(wavelength_min, wavelength_max)
plt.ylim(0)
plt.tight_layout()
