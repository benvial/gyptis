#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import pytest

from gyptis import dolfin
from gyptis.helpers import function2array, get_coords
from gyptis.sources import *

def test_pw_2D():
    lambda0 = 0.1
    theta = np.pi / 6
    mesh = dolfin.UnitSquareMesh(50, 50)
    W = dolfin.FunctionSpace(mesh, "CG", 1)
    pw = plane_wave_2D(lambda0, theta, domain=mesh)
    uproj = project(pw, W)
    uarray = function2array(uproj.real) + 1j * function2array(uproj.imag)
    x, y = get_coords(W).T
    k0 = 2 * np.pi / lambda0
    kdotx = k0 * (np.cos(theta) * x + np.sin(theta) * y)
    test = np.exp(1j * kdotx)
    err = abs(test - uarray) ** 2
    assert np.all(err < 1e-16)
    assert np.mean(err) < 1e-16


    pw, gradpw = plane_wave_2D(lambda0, theta, domain=mesh, grad=True)

    uproj = project(gradpw[1], W)


# import matplotlib.pyplot as plt
# plt.ion()
# plt.close("all")
# plt.clf()
# cm = dolfin.plot(uproj, cmap="RdBu")
# plt.colorbar(cm)


def test_pw_3D():
    theta, phi, psi = 1, 2, 3
    lambda0 = 0.1

    pw = plane_wave_3D(lambda0, theta, phi, psi)

    mesh = dolfin.UnitCubeMesh(10, 10, 10)
    W = dolfin.FunctionSpace(mesh, "CG", 1)

    x, y, z = get_coords(W).T
    k0 = 2 * np.pi / lambda0

    kx = k0 * np.sin(theta) * np.cos(phi)
    ky = k0 * np.sin(theta) * np.sin(phi)
    kz = k0 * np.cos(theta)

    cx = np.cos(psi) * np.cos(theta) * np.cos(phi) - np.sin(psi) * np.sin(phi)
    cy = np.cos(psi) * np.cos(theta) * np.sin(phi) + np.sin(psi) * np.cos(phi)
    cz = -np.cos(psi) * np.sin(theta)
    kdotx = kx * x + ky * y + kz * z
    c = np.array([cx, cy, cz])
    for i in range(3):
        uproj = project(pw[i], W)
        pw_array = function2array(uproj.real) + 1j * function2array(uproj.imag)
        pw_test = np.exp(1j * kdotx) * c[i]
        err = abs(pw_test - pw_array) ** 2
        assert np.all(err < 1e-16)
        assert np.mean(err) < 1e-16



def test_gf_2D():
    mesh = dolfin.UnitSquareMesh(50, 50)
    V = dolfin.FunctionSpace(mesh, "CG", 2)
    GF, dGF = green_function_2D(0.3,  -0.4, -0.2, degree=2, domain=mesh, grad=True,auto=False)
    proj_expr = project(GF, V)

    dGF_check = grad(GF)
    delta_grad = dot(dGF_check-dGF,dGF_check-dGF)
    error = assemble(delta_grad*dolfin.dx)
    assert abs(error) < 1e-10
    print(error)
