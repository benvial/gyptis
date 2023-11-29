#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


import pytest

from gyptis import dolfin
from gyptis.complex import assemble, div, dot, grad, project
from gyptis.sources import *
from gyptis.utils.helpers import function2array, get_coordinates


def test_pw_2d():
    lambda0 = 0.1
    theta = np.pi / 6
    mesh = dolfin.UnitSquareMesh(50, 50)
    W = dolfin.FunctionSpace(mesh, "CG", 1)
    pw = plane_wave_2d(lambda0, theta, domain=mesh)
    uproj = project(pw, W)
    uarray = function2array(uproj.real) + 1j * function2array(uproj.imag)
    x, y = get_coordinates(W).T
    k0 = 2 * np.pi / lambda0
    kdotx = -k0 * (np.sin(theta) * x + np.cos(theta) * y)
    test = np.exp(1j * kdotx)
    err = abs(test - uarray) ** 2
    assert np.all(err < 1e-16)
    assert np.mean(err) < 1e-16


def test_pw_3d():
    theta, phi, psi = 1, 2, 3
    lambda0 = 0.1

    pw = plane_wave_3d(lambda0, theta, phi, psi)

    mesh = dolfin.UnitCubeMesh(10, 10, 10)
    W = dolfin.FunctionSpace(mesh, "CG", 1)

    x, y, z = get_coordinates(W).T
    k0 = 2 * np.pi / lambda0

    kx = -k0 * np.sin(theta) * np.cos(phi)
    ky = -k0 * np.sin(theta) * np.sin(phi)
    kz = -k0 * np.cos(theta)

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


def test_gf_2d():
    mesh = dolfin.UnitSquareMesh(50, 50)
    degree = 1
    lambda0, xs, ys = 0.3, -0.1, -0.1
    GF = green_function_2d(lambda0, xs, ys, degree=degree, domain=mesh)
    k0 = 2 * np.pi / lambda0
    Helm = dot(grad(GF), grad(GF)) + k0**2 * GF * GF
    test = abs(assemble(Helm * dolfin.dx))
    print(test)
    assert test < 5e-3


def test_dipole():
    degree = 1
    mesh = dolfin.UnitSquareMesh(50, 50)
    Dipole(
        wavelength=0.1, position=(0.4, 0.2), angle=np.pi / 8, domain=mesh, degree=degree
    )


def test_gaussian_beam():
    degree = 1
    mesh = dolfin.UnitSquareMesh(50, 50)
    GaussianBeam(1, 0.4, 0.2, dim=2, domain=mesh, degree=degree)
