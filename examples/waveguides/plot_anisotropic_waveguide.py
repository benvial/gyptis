#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.0
# License: MIT
# See the documentation at gyptis.gitlab.io

"""
Anisotropic waveguide
=====================

Here we study an anisotropic square waveguide with various orientations of the optical axis.

"""

import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import gyptis as gy

##############################################################################
# We will study this benchmark and compare with results
# given in :cite:p:`Koshiba1986`.

#################################################################
# Parameters

pi = np.pi

ncore = 2.31**0.5
nclad = nsub = 2.05**0.5

pmesh = 8
wavelength = 3

wg_width = wg_thickness = 1
box_width = 1 * wg_width + 2 * wavelength
hsub = 1 * wavelength
hsup = 1 * wavelength + wg_width
lmin = wavelength / pmesh
pml_width = wavelength, wavelength


#################################################################
# Geometry

thicknesses = OrderedDict(substrate=hsub, superstrate=hsup)


geom = gy.geometry.LayeredBoxPML2D(
    box_width, thicknesses=thicknesses, pml_width=pml_width
)
sup = geom.layers["superstrate"]
sub = geom.layers["substrate"]
core = geom.add_rectangle(
    -wg_width / 2, geom.y_position["superstrate"], 0, wg_width, wg_thickness
)
out = geom.fragment(core, [sup, sub])
core = out[0]
sup, sub = out[1:]
geom.add_physical(core, "core")
geom.add_physical(sub, "substrate")
geom.add_physical(sup, "superstrate")
[geom.set_size(pml, lmin * 1) for pml in geom.pmls]
geom.set_size("superstrate", lmin / nclad)
geom.set_size("substrate", lmin / nsub)
geom.set_size("core", lmin / ncore)
geom.build()


def plot_geom(color="w"):
    geom.plot_subdomains(c=color, lw=1)


########################################
# Materials

eps_core_aniso0 = np.diag([2.31, 2.31, 2.19])

# along x axis: fig 3
# rotation: +90° around y-axis (z → x)
rot3 = Rotation.from_euler("y", 90, degrees=True)

# +45° about x axis: fig 4
# rotation: first tilt z into xy-plane, then rotate to bisect x and y
rot4 = Rotation.from_euler("xz", [90, 45], degrees=True)  # note order 'xz'


# along z axis: fig 7
# optic axis parallel to the z axis (identity)
rot7 = Rotation.from_euler("z", 0, degrees=True)

# +45° about z axis: fig 8
# rotation: +45° about x-axis (optic axis goes into yz-plane)
rot8 = Rotation.from_euler("x", 45, degrees=True)


rot_cases = {"3": rot3, "4": rot4, "7": rot7, "8": rot8}


def build_epsilon_aniso(case):
    Rmat = rot_cases[str(case)].as_matrix()
    # rotate the tensor: ε' = R ε R^T
    eps_core_aniso = Rmat @ eps_core_aniso0 @ Rmat.T
    epsilon = dict(superstrate=nclad**2, core=eps_core_aniso, substrate=nsub**2)
    return epsilon


########################################
n_eig = 8
Nwl = 40
k0t = np.linspace(3, 15, Nwl)


def run(case):
    epsilon = build_epsilon_aniso(case)

    simus = []
    effective_indices = np.zeros((Nwl, n_eig))
    for i, kt in enumerate(k0t):
        wavenumber = kt / wg_width
        k_target = wavenumber * ncore * 1.02
        simu = gy.Waveguide(
            geom,
            epsilon=epsilon,
            wavenumber=wavenumber,
            degree=(1, 1),
        )
        print(">>> Solving eigenmodes")
        t = -time.time()
        simu.eigensolve(
            n_eig=n_eig,
            target=k_target,
            tol=1e-6,
            maximum_iterations=15,
        )
        t += time.time()
        print(f"solve time: {t:.2f}s")
        print("")
        evs = simu.solution["eigenvalues"]
        modes = simu.solution["eigenvectors"]
        neff = evs / wavenumber

        # plt.plot(k0t[i] * np.ones(len(neff)), neff.real**2, ".", c="#be4848")
        # plt.pause(0.1)
        print("n_eff = ", neff)
        effective_indices[i, : len(neff)] = neff.real
        effective_indices[i, len(neff) :] = np.nan
        simus.append(simu)

    data_fig = np.loadtxt(
        f"data_aniso_fig{case}.csv", delimiter=",", skiprows=1, usecols=[0, 1]
    ).T

    plt.figure()
    plt.plot(data_fig[0], data_fig[1], ".k", ms=1, label="reference")
    plt.plot(k0t, effective_indices**2, ".", c="#be4848", label="gyptis")
    plt.xlabel(r"$k_0 t$")
    plt.ylabel(r"$(\beta/k_0)^2$")
    plt.xlim(0, 15)
    plt.ylim(2.05, 2.31)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    plt.tight_layout()
    return simus, effective_indices


########################################
# Fig 3

run(3)

########################################
# Fig 4

run(4)

########################################
# Fig 7

run(7)

########################################
# Fig 8

simus, effective_indices = run(8)


########################################
# Plot fields

neff = effective_indices[-1]
simu = simus[-1]
modes = simu.solution["eigenvectors"]

Nmodes_plot = 3

htot = geom.total_thickness
title = [r"$|E_x|$", r"$|E_y|$", r"$|E_z|$"]
for i in range(Nmodes_plot):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    E = modes[i]
    for j in range(3):
        plt.sca(ax[j])
        mappa = gy.dolfin.plot(E[j].module, cmap="inferno")
        plot_geom()
        # plt.xlim(-box_width / 2, box_width / 2)
        # plt.ylim(-htot / 2, htot / 2)
        plt.title(title[j])
        plt.colorbar(mappa)
        plt.axis("off")
    plt.suptitle(rf"$n_{{eff}}=${neff[i]:.3f}")
    plt.tight_layout()
