#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import numpy as np
import sympy as sp
from scipy.special import jv as scipy_besselj
from sympy.vector import CoordSys3D

from gyptis.complex import *

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


def plane_wave_2D(lambda0, theta, A=1, degree=1, domain=None, grad=False):
    k0 = 2 * np.pi / lambda0
    Kdir = np.array((np.cos(theta), np.sin(theta)))
    K = Kdir * k0
    K_ = vector(sp.symbols("kx, ky, 0", real=True))
    expr = A * sp.exp(1j * K_.dot(X))
    re, im = expr.as_real_imag()
    # 2D: x[2] fails!
    re, im = (p.subs(x[2], 0) for p in (re, im))
    code = [sp.printing.ccode(p) for p in (re, im)]
    dexpr = [
        dolfin.Expression(c, kx=K[0], ky=K[1], degree=degree, domain=domain)
        for c in code
    ]
    pw = Complex(*dexpr)

    if grad:
        gradre = sp.diff(re, x[0]), sp.diff(re, x[1])
        gradim = sp.diff(im, x[0]), sp.diff(im, x[1])

        code = [sp.printing.ccode(p) for p in gradre]
        dexpr_re = [
            dolfin.Expression(c, kx=K[0], ky=K[1], degree=degree, domain=domain)
            for c in code
        ]
        code = [sp.printing.ccode(p) for p in gradim]
        dexpr_im = [
            dolfin.Expression(c, kx=K[0], ky=K[1], degree=degree, domain=domain)
            for c in code
        ]

        gradpw = Complex(as_tensor(dexpr_re), as_tensor(dexpr_im))

        return pw, gradpw
    else:
        return pw


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
    prop = dolfin.Expression(
        code, kx=K[0], ky=K[1], kz=K[2], degree=degree, domain=domain
    )

    cx = np.cos(psi) * np.cos(theta) * np.cos(phi) - np.sin(psi) * np.sin(phi)
    cy = np.cos(psi) * np.cos(theta) * np.sin(phi) + np.sin(psi) * np.cos(phi)
    cz = -np.cos(psi) * np.sin(theta)

    C = dolfin.as_tensor([cx, cy, cz])
    return A * Complex(prop[0] * C, prop[1] * C)


def green_function_2D(lambda0, xs, ys, degree=1, domain=None, grad=False, auto=True):

    Xs = vector(sp.symbols("xs, ys, 0", real=True))
    k0 = sp.symbols("k0", real=True)
    Xshift = X - Xs
    rho = sp.sqrt(Xshift.dot(Xshift))
    rho = rho.subs(x[2], 0)
    kr = k0 * rho
    k0_ = 2 * np.pi / lambda0
    KR = dolfin.Expression(
        sp.printing.ccode(kr), k0=k0_, xs=xs, ys=ys, degree=degree, domain=domain
    )

    GF = -1 / 4 * Complex(dolfin.bessel_Y(0, KR), dolfin.bessel_J(0, KR))

    if grad:
        if auto:
            gradGF = grad(GF)
        else:
            gradre = dolfin.bessel_Y(-1, KR) - dolfin.bessel_Y(1, KR)
            gradim = dolfin.bessel_J(-1, KR) - dolfin.bessel_J(1, KR)
            coeff_x = Xshift.dot(N.i) * (-k0 / (8 * rho))
            coeff_y = Xshift.dot(N.j) * (-k0 / (8 * rho))
            coeff = coeff_x, coeff_y
            Coeff = [
                dolfin.Expression(
                    sp.printing.ccode(co),
                    k0=k0_,
                    xs=xs,
                    ys=ys,
                    degree=degree,
                    domain=domain,
                )
                for co in coeff
            ]
            C = dolfin.as_tensor(Coeff)
            gradGF = Complex(gradre * C, gradim * C)

        return GF, gradGF
    else:
        return GF
