#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
Introduction to electromagnetic scattering: tutorial
https://www.osapublishing.org/josaa/fulltext.cfm?uri=josaa-35-1-163&id=380136
"""

import numpy as np
from dolfin import MPI
from mie import get_cross_sections_analytical

from gyptis import BoxPML, PlaneWave, Scattering, c, dolfin, epsilon_0, mu_0
from gyptis.complex import *
from gyptis.utils.helpers import mpi_print

plot_scs = True
if plot_scs:
    from gyptis.plot import *

    plt.ion()
    plt.close("all")
rank = MPI.rank(MPI.comm_world)

# eps_sphere = 4
# a = 0.25
eps_sphere = (5 - 0.4 * 1j) ** 2
a = 0.1

eps_bg = 1
S_sphere = a**2 * np.pi
circ = 2 * np.pi * a

shared_datadir = "../../tests/data"
## needed for surface integral in parallel (mpi)
dolfin.parameters["ghost_mode"] = "shared_facet"
dolfin.parameters["form_compiler"]["quadrature_degree"] = 3

scs_file = shared_datadir + "/sphere_diel.csv"
benchmark = np.loadtxt(scs_file, delimiter=",")

GAMMA = benchmark[:, 0]
CS_NORMA_REF = benchmark[:, 1]


Gamma = np.linspace(0.1, 5, 500)
Gamma = np.linspace(0.1, 2, 500)
lambda0 = circ / Gamma
ks = 2 * np.pi / lambda0

CSANA, CEANA, CAANA = get_cross_sections_analytical(ks, a, eps_sphere, eps_bg)
CSANA_NORMA_MIE = CSANA / S_sphere
CEANA_NORMA_MIE = CEANA / S_sphere
CAANA_NORMA_MIE = CAANA / S_sphere

CAANA_NORMA_MIE = CEANA_NORMA_MIE - CSANA_NORMA_MIE

if plot_scs and rank == 0:
    plt.figure(figsize=(4, 3))
    # plt.plot(GAMMA, CS_NORMA_REF, "--", c="#525252", label="Ref")
    plt.plot(Gamma, CSANA_NORMA_MIE, c="#545cc7", label="SCS Mie")
    plt.plot(Gamma, CEANA_NORMA_MIE, c="#54c777", label="SCE Mie")
    plt.plot(Gamma, CAANA_NORMA_MIE, c="#c79c54", label="SCA Mie")

    # arch = np.load("cross_sections_gyptis.npz", allow_pickle=True)
    # Gamma_ref = arch["Gamma"]
    # SCSN = arch["SCSN"].tolist()
    # plt.plot(Gamma_ref, SCSN["scattering"], "o", c="#545cc7", label="SCS gyptis")
    # plt.plot(Gamma_ref, SCSN["extinction"], "o", c="#54c777", label="SCE gyptis")
    # plt.plot(Gamma_ref, SCSN["absorption"], "o", c="#c79c54", label="SCA gyptis")
    #
    #
    #
    plt.ylim(0)

    plt.xlabel(r"circumfenrence/wavelength $k_0 a$")
    plt.ylabel(r"normalized scattering cross section $\sigma_s / S$")
    plt.legend()
    plt.tight_layout()


def build_geometry(pmesh):

    pmesh_scatt = 1 * pmesh

    b = a * 2 * 1.2
    box_size = (b, b, b)
    pml_width = (lambda0, lambda0, lambda0)

    Rcalc = (min(box_size) / 2 + a) / 2

    # Rcalc = 0
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
        # geom.add_physical(surfs, "calc_bnds",2)
    else:
        surfs = geom.get_boundaries("box")[:-1]
        names = ["-x", "-y", "+z", "+y", "-z", "+x"]
        for surface, name in zip(surfs, names):
            geom.add_physical(surface, name, 2)

    smin = a / 3
    s = min(lambda0 / pmesh, smin)

    smin_pml = lambda0 / (0.7 * pmesh)

    for coord in ["x", "y", "z", "xy", "xz", "yz", "xyz"]:
        geom.set_mesh_size({"pml" + coord: smin_pml})

    geom.set_size(box, s)
    geom.set_size(surfs, s, dim=2)
    s = min(lambda0 / (eps_sphere.real**0.5 * pmesh_scatt), smin)
    # s =lambda0 / (eps_sphere ** 0.5 * pmesh_scatt)
    geom.set_size(sphere, s)
    geom.build()
    return geom


def compute_scs(lambda0, pmesh=2, degree=1):

    # Rcalc=0.9*b/2
    mpi_print("####################################")
    mpi_print("MESHING")
    mpi_print("####################################")

    geom = build_geometry(pmesh)

    epsilon = dict(sphere=eps_sphere, box=eps_bg)
    mu = dict(sphere=1, box=1)

    mpi_print("####################################")
    mpi_print("COMPUTING EM PROBLEM")
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
    mpi_print("####################################")
    mpi_print("COMPUTING CROSS SECTIONS")
    mpi_print("####################################")

    # scs = scatt.scattering_cross_section()
    Sigma_s = scatt.scattering_cross_section()
    Sigma_e = scatt.extinction_cross_section()
    Sigma_a = scatt.absorption_cross_section()
    return Sigma_s, Sigma_e, Sigma_a, scatt


degree = 1
pmesh = 10

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

    OT = Sigma_e_norm - Sigma_s_norm - Sigma_a_norm
    mpi_print(f"error optical theorem: {np.abs(OT)}")

    if plot_scs and rank == 0:
        plt.plot(gamma, Sigma_s_norm, "o", c="#545cc7", label="gyptis")
        plt.plot(gamma, Sigma_e_norm, "o", c="#54c777", label="gyptis")
        plt.plot(gamma, Sigma_a_norm, "o", c="#c79c54", label="gyptis")
        # plt.plot(P, SCSN, "-o", c="#c64545", label="gyptis")
        plt.pause(0.1)

if rank == 0:
    np.savez("cross_sections.npz", SCSN=SCSN, Gamma=Gamma)

#
#
# PMESH = [3, 5, 7, 10]
# # PMESH = [7]
# SCSN = []
# P=[]
# for pmesh in PMESH:
#     lambda0 = circ / gamma
#     Sigma_s = compute_scs(lambda0, pmesh=pmesh,degree=degree)
#     Sigma_s_norm = Sigma_s / S_sphere
#     mpi_print(Sigma_s_norm)
#     SCSN.append(Sigma_s_norm)
#     P.append(pmesh)
#
#
#     if rank ==0:
#         # plt.plot(pmesh, Sigma_s_norm, "o", c="#c64545", label="gyptis")
#         plt.plot(P, SCSN, "-o", c="#c64545", label="gyptis")
#         plt.pause(0.1)


#
#
#
# from gyptis.utils.helpers import project_iterative
# u = scatt.solution["total"]
# V = scatt.formulation.real_function_space
#
# Vproj = dolfin.FunctionSpace(V.mesh(),"CG",degree)
# uplt = project_iterative(u.real[0], Vproj)
#
# dolfin.File("uplt.pvd") << uplt
#
# import os
# os.system("paraview uplt.pvd")
#
# #
# # from fenics_plotly import plot as fpplot
# #
# # out = fpplot(uplt, show=True, colorscale="rdbu_r", wireframe=False,opacity=0.1)
# #
# # fig_field = out.figure
# # camera = dict(up=dict(x=0, y=1, z=0), eye=dict(x=0.0, y=0.0, z=25))
# #
# # fig_field.update_layout(
# #     scene_camera=camera,
# #     autosize=False,
# #     # width=500,
# #     height=700,
# #     title=dict(text="field map"),
# # )
#
# # from vedo import embedWindow
# # import vedo.dolfin as vdf
# # embedWindow(False)
# #
# # # either
# #
# # # out = vdf.plot(uplt)
# # # vdf.show(new=True)
# #
# # plt =vdf.plot(uplt,
# #      style=0,
# #      elevation=-3, # move camera a bit
# #      azimuth=1,
# #      lighting='plastic',
# #      cmap='rainbow', # the color map style
# #      alpha=1,        # transparency of the mesh
# #      lw=0.1,         # linewidth of mesh
# #      interactive=1)
# #
# #
# # msh = plt.actors[0]
# # msh.cutWithPlane() # etc
# # plt.show()
