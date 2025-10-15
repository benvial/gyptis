#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.3
# License: MIT
# See the documentation at gyptis.gitlab.io


"""
A scattering simulation in 3D
-----------------------------

In this example we will study the scattering of a plane wave by a nanosphere
"""


############################################################################
# Reference results are taken from :cite:p:`Frezza2018`.


plot_scs = True
if plot_scs:
    import matplotlib.pyplot as plt

    from gyptis.plot import *

    plt.ion()
    plt.close("all")

import logging

import numpy as np
from dolfin import MPI
from mie import get_cross_sections_analytical

from gyptis import BoxPML, PlaneWave, Scattering, c, dolfin, epsilon_0, mu_0
from gyptis.complex import *
from gyptis.utils import mpi_print, set_log_level

# set_log_level(logging.DEBUG)


rank = MPI.rank(MPI.comm_world)

eps_sphere = (5 - 0.4 * 1j) ** 2
a = 0.1

eps_bg = 1
S_sphere = a**2 * np.pi
circ = 2 * np.pi * a

shared_datadir = "../../tests/data"
# dolfin.parameters["form_compiler"]["quadrature_degree"] = 3

scs_file = f"{shared_datadir}/sphere_diel.csv"
benchmark = np.loadtxt(scs_file, delimiter=",")

GAMMA = benchmark[:, 0]
CS_NORMA_REF = benchmark[:, 1]


Gamma = np.linspace(0.1, 2, 500)
lambda0 = circ / Gamma
ks = 2 * np.pi / lambda0

CSANA, CEANA, CAANA = get_cross_sections_analytical(ks, a, eps_sphere, eps_bg)
CSANA_NORMA_MIE = CSANA / S_sphere
CEANA_NORMA_MIE = CEANA / S_sphere
CAANA_NORMA_MIE = CAANA / S_sphere


if plot_scs and rank == 0:
    plt.figure()
    plt.plot(Gamma, CSANA_NORMA_MIE, c="#545cc7", label="SCS Mie")
    plt.plot(Gamma, CEANA_NORMA_MIE, c="#54c777", label="SCE Mie")
    plt.plot(Gamma, CAANA_NORMA_MIE, c="#c79c54", label="SCA Mie")
    plt.ylim(0)
    plt.xlabel(r"circumfenrence/wavelength $k_0 a$")
    plt.ylabel(r"normalized scattering cross section $\sigma_s / S$")
    plt.legend()
    plt.tight_layout()
    CS_NORMA_REF

    plt.plot(GAMMA, CS_NORMA_REF, c="#4c4c4c", label="SCS reference")

    sdds


def build_geometry(pmesh):
    pmesh_scatt = 1 * pmesh
    b = a * 2 * 1.2
    box_size = (b, b, b)
    pml_width = (lambda0, lambda0, lambda0)
    Rcalc = (min(box_size) / 2 + a) / 2
    geom = BoxPML(3, box_size=box_size, pml_width=pml_width, Rcalc=Rcalc)

    box = geom.box
    sphere = geom.add_sphere(0, 0, 0, a)
    if Rcalc > 0:
        sphere, *box = geom.fragment(sphere, box)
    else:
        sphere, box = geom.fragment(sphere, box)
    geom.add_physical(box, "box")
    geom.add_physical(sphere, "sphere")

    if Rcalc > 0:
        surfs = geom.get_boundaries("box")[-1]
    else:
        surfs = geom.get_boundaries("box")[:-1]
        names = ["-x", "-y", "+z", "+y", "-z", "+x"]
        for surface, name in zip(surfs, names):
            geom.add_physical(surface, name, 2)

    smin = a / 3
    s = min(lambda0 / pmesh, smin)

    smin_pml = lambda0 / (0.7 * pmesh)
    for coord in ["x", "y", "z", "xy", "xz", "yz", "xyz"]:
        geom.set_mesh_size({f"pml{coord}": smin_pml})

    geom.set_size(box, s)
    geom.set_size(surfs, s, dim=2)
    s = min(lambda0 / (eps_sphere.real**0.5 * pmesh_scatt), smin)
    geom.set_size(sphere, s)
    geom.build()
    return geom


def compute_scs(lambda0, pmesh=2, degree=1):
    mpi_print("MESHING")
    mpi_print("####################################")

    geom = build_geometry(pmesh)

    epsilon = dict(sphere=eps_sphere, box=eps_bg)
    mu = dict(sphere=1, box=1)

    mpi_print("SOLVING EM PROBLEM")
    mpi_print("####################################")
    pw = PlaneWave(
        wavelength=lambda0, angle=(0, 0, 0), dim=3, domain=geom.mesh, degree=degree
    )
    bcs = {}
    scatt = Scattering(
        geom,
        epsilon,
        mu,
        pw,
        boundary_conditions=bcs,
        degree=degree,
    )

    scatt.solve()
    mpi_print("COMPUTING CROSS SECTIONS")
    mpi_print("####################################")

    Sigma_s = scatt.scattering_cross_section()
    Sigma_e = scatt.extinction_cross_section()
    Sigma_a = scatt.absorption_cross_section()
    return Sigma_s, Sigma_e, Sigma_a, scatt


degree = 2
pmesh = 1

SCSN = []
P = []
Gamma = np.linspace(0.25, 2, 100)

Gamma = [0.3, 0.7, 1, 1.25]


SCSN = dict(scattering=[], extinction=[], absorption=[])
for gamma in Gamma:
    lambda0 = circ / gamma
    Sigma_s, Sigma_e, Sigma_a, scatt = compute_scs(lambda0, pmesh=pmesh, degree=degree)
    Sigma_s_norm = Sigma_s / S_sphere
    Sigma_e_norm = Sigma_e / S_sphere
    Sigma_a_norm = Sigma_a / S_sphere
    mpi_print(Sigma_s_norm)
    mpi_print(Sigma_e_norm)
    mpi_print(Sigma_a_norm)
    SCSN["scattering"].append(Sigma_s_norm)
    SCSN["extinction"].append(Sigma_e_norm)
    SCSN["absorption"].append(Sigma_a_norm)
    P.append(gamma)

    k = 2 * np.pi / lambda0
    CSANA, CEANA, CAANA = get_cross_sections_analytical(k, a, eps_sphere, eps_bg)
    Sigma_e_norm_mie = CEANA / S_sphere
    Sigma_s_norm_mie = CSANA / S_sphere
    Sigma_a_norm_mie = CAANA / S_sphere

    OT = Sigma_e_norm - Sigma_s_norm - Sigma_a_norm
    mpi_print(f"error optical theorem: {OT}")
    mpi_print(f"relative error SCS: {100*np.abs(1-Sigma_e_norm/Sigma_e_norm_mie)}%")
    mpi_print(f"relative error ECS: {100*np.abs(1-Sigma_s_norm/Sigma_s_norm_mie)}%")
    mpi_print(f"relative error ACS: {100*np.abs(1-Sigma_a_norm/Sigma_a_norm_mie)}%")

    if plot_scs and rank == 0:
        plt.plot(gamma, Sigma_s_norm, "o", c="#545cc7", label="gyptis")
        plt.plot(gamma, Sigma_e_norm, "o", c="#54c777", label="gyptis")
        plt.plot(gamma, Sigma_a_norm, "o", c="#c79c54", label="gyptis")
        plt.pause(0.1)

if rank == 0:
    np.savez("cross_sections.npz", SCSN=SCSN, Gamma=Gamma)
