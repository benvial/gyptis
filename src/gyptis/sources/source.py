#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

"""
Sources.
"""

import numpy as np
import sympy as sp
from dolfin import Constant as ConstantRe
from scipy.constants import c, epsilon_0, mu_0
from sympy.vector import CoordSys3D

from ..complex import Complex, Constant, as_tensor, as_vector, dolfin, dot, grad

_COORD = CoordSys3D("N")


def vector(components):
    return (
        components[0] * _COORD.i + components[1] * _COORD.j + components[2] * _COORD.k
    )


x = sp.symbols("x[0] x[1] x[2]", real=True)
X = vector(np.array(x))


def expression2complex_2d(expr, **kwargs):
    re, im = (p.subs(x[2], 0) for p in expr.as_real_imag())
    dexpr = [dolfin.Expression(sp.printing.ccode(p), **kwargs) for p in (re, im)]
    return Complex(*dexpr)


def green_function_2d(wavelength, xs, ys, amplitude=1, degree=1, domain=None):
    Xs = vector(sp.symbols("xs, ys, 0", real=True))
    k0 = sp.symbols("k0", real=True)
    Xshift = X - Xs
    rho = sp.sqrt(Xshift.dot(Xshift))
    rho = rho.subs(x[2], 0)
    kr = k0 * rho
    k0_ = 2 * np.pi / wavelength
    KR = dolfin.Expression(
        sp.printing.ccode(kr), k0=k0_, xs=xs, ys=ys, degree=degree, domain=domain
    )
    return (
        -1
        / 4
        * Complex(dolfin.bessel_Y(0, KR), dolfin.bessel_J(0, KR))
        * Constant(amplitude)
    )


class Source:
    def __init__(self, wavelength, dim, degree=1, domain=None):
        self.wavelength = wavelength
        self.dim = dim
        self.degree = degree
        self.domain = domain

    @property
    def wavenumber(self):
        return 2 * np.pi / self.wavelength

    @property
    def pulsation(self):
        return self.wavenumber * c

    @property
    def frequency(self):
        return c / self.wavelength
