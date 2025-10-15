#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.0
# License: MIT
# See the documentation at gyptis.gitlab.io

"""
A simple waveguide
==================


"""


import matplotlib.pyplot as plt

plt.ion()
plt.close("all")
import dolfin
import numpy as np

import gyptis as gy
from gyptis.models import FibersConical

#################################################################
# Parameters

pi = np.pi
neff = 1.27627404
lambda_ = 1.5

beta = neff * 2 * pi / lambda_
k_target = 2 * pi / lambda_

Nx = Ny = 1  # must be odd
pmesh = 20

wavelength = 1
period = 1

L = 1
lmin = wavelength / pmesh


eps1 = 2.25
eps2 = 1


#################################################################
# Geometry

geom = gy.Geometry(dim=2)
box1 = geom.add_square(-L / 2, -L / 2, 0, L)
box2 = geom.add_square(-L / 2, -L / 2, 0, L / 2)

box1, box2 = geom.fragment(box1, box2)

# box1:small, box2: large

geom.add_physical(box1, "box1")
geom.add_physical(box2, "box2")
geom.set_size("box1", lmin / eps1**0.5)
geom.set_size("box2", lmin / eps2**0.5)
geom.build()

# geom.plot_mesh()


########################################
# Materials

epsilon = dict(box1=eps1, box2=eps2)

# eps_aniso = np.diag([4, 4, 4])
# epsilon = dict(box=1, fiber=eps_aniso)


########################################


simu = FibersConical(
    geom, epsilon=epsilon, beta=beta, modal=True, degree=(2, 2), pmls=False
)

simu.eigensolve(
    n_eig=1,
    target=k_target,
    tol=1e-6,
)
evs = simu.solution["eigenvalues"]
modes = simu.solution["eigenvectors"]
print(evs)

title = [r"$E_x$", r"$E_y$", r"$E_z$"]
for i in range(len(evs)):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    H = modes[i]
    for j in range(3):
        plt.sca(ax[j])
        mappa = gy.dolfin.plot(H[j].real, cmap="RdBu_r")
        # geom.plot()
        plt.xlim(-L / 2, L / 2)
        plt.ylim(-L / 2, L / 2)
        plt.title(title[j])
        plt.colorbar(mappa)
        plt.axis("off")
    plt.suptitle(rf"k={evs[i]:.5f}")
    plt.tight_layout()


print("lambda = ", 2 * pi / evs)
