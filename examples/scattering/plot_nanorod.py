#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
Nanorods
========

Scattering of a Gaussian beam wave by an array of metallic thin wires
"""

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy

pmesh = 6
degree = 2

R = 20
d = 150
N = 40
wavelength = 750
lbox_x = (N) * d + 4 * wavelength
lbox_y = 6 * wavelength


def build_geo():
    lmin = wavelength / pmesh
    geom = gy.BoxPML(
        dim=2,
        box_size=(lbox_x, lbox_y),
        box_center=(0, 0),
        pml_width=(wavelength, wavelength),
    )
    box = geom.box
    rods = [geom.add_circle(-N / 2 * d + i * d, 0, 0, R) for i in range(N)]
    *rods, box = geom.fragment(box, rods)
    geom.add_physical(box, "box")
    geom.add_physical(rods, "rods")
    [geom.set_size(pml, lmin * 0.7) for pml in geom.pmls]
    geom.set_size("box", lmin)
    geom.set_size("rods", lmin)
    geom.build()
    return geom


def run(geom, polarization):
    pw = gy.GaussianBeam(
        wavelength=wavelength,
        angle=gy.pi,
        waist=wavelength,
        position=(-N / 2 * d, 0),
        dim=2,
        domain=geom.mesh,
        degree=degree,
    )

    omega = 2 * gy.pi * gy.c / (wavelength * 1e-6)
    omega_p = 21**0.5 * omega
    eps_rod = 1 - omega_p**2 / omega**2

    epsilon = dict(box=1, rods=eps_rod)
    mu = dict(box=1, rods=1)
    s = gy.Scattering(
        geom,
        epsilon,
        mu,
        pw,
        degree=degree,
        polarization=polarization,
    )

    s.solve()

    return s


def plot_solution(s, title):
    plt.figure(figsize=(4, 1.8))
    s.plot_field(type="module", cmap="inferno")
    ax = plt.gca()
    for i in range(N):
        cir = plt.Circle((-N / 2 * d + i * d, 0), R, lw=0.3, color="w", fill=False)
        ax.add_patch(cir)
    plt.xlabel("x (nm)")
    plt.ylabel("y (nm)")
    plt.title(title)
    plt.tight_layout()


##############################################################################
# Build the geometry

geom = build_geo()


##############################################################################
# TM polarization

sTM = run(geom, "TM")
plot_solution(sTM, "Electric field norm")


##############################################################################
# TE polarization

sTE = run(geom, "TE")
plot_solution(sTE, "Magnetic field norm")
