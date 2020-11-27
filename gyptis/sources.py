#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import dolfin as df
import numpy as np
import sympy as sp
from sympy.vector import CoordSys3D

from gyptis.complex import Complex, project

N = CoordSys3D("N")
# v1 = 2 * N.i + 3 * N.j - N.k
# v2 = N.i - 4 * N.j + N.k
# v1.dot(v2)
# v1.cross(v2)
# # Alternately, can also do
# v1 & v2
# v1 ^ v2
vector = lambda x: x[0] * N.i + x[1] * N.j + x[2] * N.k

x = sp.symbols("x[0] x[1] x[2]", real=True)
X = vector(np.array(x))


def plane_wave_2D(lambda0, theta, A=1, degree=1, domain=None):
    k0 = 2 * np.pi / lambda0
    Kdir = np.array((np.cos(theta), np.sin(theta)))
    K = Kdir * k0
    K_ = vector(sp.symbols("kx, ky, 0", real=True))
    expr = sp.exp(1j * K_.dot(X))
    re, im = expr.as_real_imag()
    # 2D: x[2] fails!
    re, im = (p.subs(x[2], 0) for p in (re, im))
    code = [sp.printing.ccode(p) for p in (re, im)]
    dexpr = df.Expression(code, kx=K[0], ky=K[1], degree=degree, domain=domain)
    return A * Complex(*dexpr)


def plane_wave_3D(lambda0, theta, phi, psi, A=1, degree=1, domain=None):

    k0 = 2 * np.pi / lambda0
    Kdir = np.array(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        )
    )
    K = Kdir * k0
    K_ = vector(sp.symbols("kx, ky, kz", real=True))

    Propp = sp.exp(1j * K_.dot(X))
    re, im = Propp.as_real_imag()
    code = [sp.printing.ccode(p) for p in (re, im)]
    prop = df.Expression(code, kx=K[0], ky=K[1], kz=K[2], degree=degree, domain=domain)

    cx = np.cos(psi) * np.cos(theta) * np.cos(phi) - np.sin(psi) * np.sin(phi)
    cy = np.cos(psi) * np.cos(theta) * np.sin(phi) + np.sin(psi) * np.cos(phi)
    cz = -np.cos(psi) * np.sin(theta)

    C = df.as_tensor([cx, cy, cz])
    return A * Complex(prop[0] * C, prop[1] * C)
