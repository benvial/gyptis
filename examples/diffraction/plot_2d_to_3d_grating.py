#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
2D/3D grating comparison
========================

Check results from 3D simulation against 2D.
"""
# sphinx_gallery_thumbnail_number = 2


import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pyvista

import gyptis as gy

# from ufl.algorithms.compute_form_data \
#     import estimate_total_polynomial_degree

# estimate_total_polynomial_degree(simu2D.formulation.weak)
# estimate_total_polynomial_degree(simu3D.formulation.weak)

##############################################################################
# We first define some parameters

lambda0 = 1
dy = 1.8
dx = dy / 20
l_pillar = dy / 2
h = lambda0
theta0 = 0
phi0 = 0
eps_diel = 4
eps_pillar = 3 - 0.2j
degree = 2
pmesh = 8
nord = 3


gy.dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

##############################################################################
# The thicknesses of the different layers are specified with an
# ``OrderedDict`` object **from bottom to top**:

thicknesses = OrderedDict(
    {
        "pml_bottom": lambda0,
        "substrate": lambda0,
        "groove": h,
        "superstrate": lambda0,
        "pml_top": lambda0,
    }
)

##############################################################################
# Here we set the mesh refinement parameters, in order to be able to have
# ``parmesh`` cells per wavelength of the field inside each subdomain

mesh_param = dict(
    {
        "pml_bottom": 0.7 * pmesh * eps_diel**0.5,
        "substrate": pmesh * eps_diel**0.5,
        "groove": pmesh * eps_diel**0.5,
        "rod": pmesh * abs(eps_pillar.real) ** 0.5,
        "superstrate": pmesh,
        "pml_top": 0.7 * pmesh,
    }
)

##############################################################################
# Let's create the geometry using the :class:`~gyptis.Layered`
# class:


def build_geometry(dim):
    if dim == 2:
        geom = gy.Layered(2, dy, thicknesses)
        y0 = geom.y_position["groove"]
        pillar = geom.add_rectangle(-l_pillar / 2, y0, 0, l_pillar, h)
    else:
        geom = gy.Layered(3, (dx, dy), thicknesses)
        z0 = geom.z_position["groove"]
        pillar = geom.add_box(-dx / 2, -l_pillar / 2, z0, dx, l_pillar, h)
    groove = geom.layers["groove"]
    sub = geom.layers["substrate"]
    sup = geom.layers["superstrate"]
    out = geom.fragment([sub, sup, groove], pillar)
    sub = out[0]
    sup = out[1]
    pillar = out[2]
    groove = out[3:]
    geom.add_physical(pillar, "rod")
    geom.add_physical(groove, "groove")
    geom.add_physical(sub, "substrate")
    geom.add_physical(sup, "superstrate")
    mesh_size = {d: lambda0 / param for d, param in mesh_param.items()}
    geom.set_mesh_size(mesh_size)
    geom.build(interactive=False)
    return geom


######################################################################
# Now we can create an instance of the simulation class
# :class:`~gyptis.Grating`,


def init_simu(geom, polarization):
    dim = geom.dim
    mu = {d: 1 for d in geom.domains}
    epsilon = {d: 1 for d in geom.domains}
    epsilon["rod"] = eps_pillar
    epsilon["substrate"] = eps_diel
    if dim == 2:
        pw = gy.PlaneWave(lambda0, theta0, dim=2)
        grating = gy.Grating(
            geom, epsilon, mu, source=pw, polarization=polarization, degree=degree
        )
    else:
        psi0 = 0 if polarization == "TM" else gy.pi / 2
        pw = gy.PlaneWave(lambda0, (theta0, phi0, psi0), dim=3, degree=degree)
        grating = gy.Grating(
            geom, epsilon, mu, source=pw, degree=degree, periodic_map_tol=1e-12
        )
    return grating


######################################################################
# Main function


def run(polarization):
    t2Dmesh = -time.time()
    geom2D = build_geometry(2)
    t2Dmesh += time.time()
    simu2D = init_simu(geom2D, polarization)

    t2Dsolve = -time.time()
    simu2D.solve()
    t2Dsolve += time.time()

    t2Deff = -time.time()
    effs2D = simu2D.diffraction_efficiencies(nord, orders=True)
    t2Deff += time.time()

    t3Dmesh = -time.time()
    geom3D = build_geometry(3)
    t3Dmesh += time.time()
    simu3D = init_simu(geom3D, polarization)

    t3Dsolve = -time.time()
    simu3D.solve()
    t3Dsolve += time.time()

    t3Deff = -time.time()
    effs3D = simu3D.diffraction_efficiencies(nord, orders=True)
    t3Deff += time.time()

    T2D = np.array(effs2D["T"])
    T3D = np.array(effs3D["T"])[nord]
    R2D = np.array(effs2D["R"])
    R3D = np.array(effs3D["R"])[nord]
    Q2D = effs2D["Q"]
    Q3D = effs3D["Q"]
    B2D = effs2D["B"]
    B3D = effs3D["B"]

    print()
    print("=========================================")
    print(f"             {polarization} polarization")
    print("=========================================")
    k = 0
    print()
    print("                    2D           3D      ")
    print("-----------------------------------------")

    print("Transmission")
    for i in range(-nord, nord + 1):
        i1 = f"+{i}" if i > 0 else i
        spacex = "" if i != 0 else " "
        print(f"T({i1}) {spacex}            {T2D[k]:.5f}      {T3D[k]:.5f}  ")
        k += 1
    print(f"T                 {sum(T2D):.5f}      {sum(T3D):.5f}  ")
    k = 2
    print()
    print("Reflection")
    for i in range(-1, 2):
        i1 = f"+{i}" if i > 0 else i
        spacex = "" if i != 0 else " "
        print(f"R({i1}) {spacex}            {R2D[k]:.5f}      {R3D[k]:.5f}  ")
        k += 1
    print(f"R                 {sum(R2D):.5f}      {sum(R3D):.5f}  ")
    print()
    print("Absorption")
    print(f"Q                 {Q2D:.5f}      {Q3D:.5f}  ")
    print()
    print("Energy balance")
    print(f"B                 {B2D:.5f}      {B3D:.5f}  ")
    print()
    print("Number of DOF")
    print(f"                   {simu2D.ndof}        {simu3D.ndof}  ")
    print()
    print("CPU time")
    print(f"mesh               {t2Dmesh:.3f}s       {t3Dmesh:.3f}s  ")
    print(f"solve              {t2Dsolve:.3f}s      {t3Dsolve:.3f}s  ")
    print(f"efficiencies       {t2Deff:.3f}s       {t3Deff:.3f}s  ")

    utot = simu2D.solution["total"].real
    gy.plot(utot)
    plt.title(f"2D, {polarization}")

    V = gy.dolfin.FunctionSpace(geom3D.mesh, "CG", 1)
    E = simu3D.solution["total"]
    H = simu3D.formulation.get_magnetic_field(E)
    comp = 0
    Z0 = (gy.mu_0 / gy.epsilon_0) ** 0.5
    field = Z0 * H if polarization == "TE" else E
    gy.dolfin.File("reE.pvd") << gy.project_iterative(field[comp].real, V)
    reader = pyvista.get_reader("reE.pvd")
    mesh = reader.read()
    pl = pyvista.Plotter()
    _ = pl.add_mesh(
        mesh,
        cmap="RdBu_r",
        interpolate_before_map=True,
        scalar_bar_args={"title": "Re E", "vertical": True},
    )
    pl.add_title(f"3D, {polarization}")
    pl.view_yz()
    pl.show()


######################################################################
# TE

run("TE")


######################################################################
# TM

run("TM")
