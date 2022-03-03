#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


"""
Scattering Matrix calculations for a scatterer.

See:
Demésy, G., Auger, J.-C. & Stout, B. Scattering matrix of arbitrarily shaped
objects: combining finite elements and vector partial waves.
J. Opt. Soc. Am. A, JOSAA 35, 1401–1409 (2018)
"""

import scipy.special as sp

from .. import dolfin as df
from .. import pi
from ..complex import Complex, Constant, j, vector
from .source import Source


def sqrt_safe(x):
    x = Complex(x, 0)
    t = x.phase
    return df.sqrt(x.module) * Complex(df.cos(t / 2), df.sin(t / 2))


def sph_bessel_J(nu, f):
    return df.sqrt(pi / (2 * f)) * df.bessel_J(nu + 0.5, f)


def sph_bessel_Y(nu, f):
    return df.sqrt(pi / (2 * f)) * df.bessel_Y(nu + 0.5, f)


def sph_bessel_H1(nu, f):
    return Complex(sph_bessel_J(nu, f), sph_bessel_Y(nu, f))


def sph_bessel_H2(nu, f):
    return Complex(sph_bessel_J(nu, f), -sph_bessel_Y(nu, f))


def ricatti_bessel_J(nu, f):
    return f * sph_bessel_J(nu, f)


def ricatti_bessel_Y(nu, f):
    return f * sph_bessel_Y(nu, f)


def ricatti_bessel_H1(nu, f):
    return Complex(ricatti_bessel_J(nu, f), ricatti_bessel_Y(nu, f))


def ricatti_bessel_H2(nu, f):
    return Complex(ricatti_bessel_J(nu, f), -ricatti_bessel_Y(nu, f))


def sph_jn(nu, f):
    return (pi / (2 * f)) ** 0.5 * sp.jv(nu + 0.5, f)


def sph_yn(nu, f):
    return (pi / (2 * f)) ** 0.5 * sp.yv(nu + 0.5, f)


def sph_hn1(nu, f):
    return sph_jn(nu, f) + 1j * sph_yn(nu, f)


def sph_hn2(nu, f):
    return sph_jn(nu, f) - 1j * sph_yn(nu, f)


def rb_jn(nu, f):
    return f * sph_jn(nu, f)


def rb_yn(nu, f):
    return f * sph_yn(nu, f)


def rb_hn1(nu, f):
    return rb_jn(nu, f) + 1j * rb_yn(nu, f)


def rb_hn2(nu, f):
    return rb_jn(nu, f) - 1j * rb_yn(nu, f)


def nm2p(n, m):
    return n * (n + 1) - m


def p2nm(p):
    n = int(p ** 0.5)
    m = n * (n + 1) - p
    return n, m


# Recurrence relations for scalar spherical harmonics


def P(x, n, m, y=None):
    y = y or sqrt_safe(1 - x ** 2)
    if m < 0:
        return (-1) ** (-m) * P(x, n, -m)
    elif n == m == 0:
        return 1 / (4 * pi) ** 0.5
    elif n == m:
        return -(((2 * n + 1) / (2 * n)) ** 0.5) * y * P(x, n - 1, m - 1)
    elif m == (n - 1):
        return x * (2 * n + 1) ** 0.5 * P(x, n - 1, n - 1)
    else:
        return (((2 * n + 1) / (n ** 2 - m ** 2)) ** 0.5) * (
            (2 * n - 1) ** 0.5 * x * P(x, n - 1, m)
            - (((n - 1) ** 2 - m ** 2) / (2 * n - 3)) ** 0.5 * P(x, n - 2, m)
        )


def u(x, n, m, y=None):
    y = y or sqrt_safe(1 - x ** 2)
    if m < 0:
        return (-1) ** (-m + 1) * u(x, n, -m)
    elif m == 0:
        return 0
    elif m == 1:
        return -1 / 4 * (3 / pi) ** 0.5
    elif n == m:
        return (
            -((n * (2 * n + 1) / (2 * (n + 1) * (n - 1))) ** 0.5)
            * y
            * u(x, n - 1, n - 1)
        )
    elif m == (n - 1):
        return ((2 * n + 1) * (n - 1) / (n + 1)) ** 0.5 * x * u(x, n - 1, n - 1)
    else:
        return ((2 * n + 1) * (n - 1) / ((n + 1) * (n ** 2 - m ** 2))) ** 0.5 * (
            x * (2 * n - 1) ** 0.5 * u(x, n - 1, m)
            - ((n - 2) * ((n - 1) ** 2 - m ** 2) / (n * (2 * n - 3))) ** 0.5
            * u(x, n - 2, m)
        )


def s(x, n, m, y=None):
    y = y or sqrt_safe(1 - x ** 2)
    if m < 0:
        return (-1) ** (-m) * s(x, n, -m)
    elif n == m:
        return x * u(x, n, m)
    else:
        return 1 / (m + 1) * ((n + m + 1) * (n - m)) ** 0.5 * y * u(
            x, n, m + 1
        ) + x * u(x, n, m)


class SphericalSource(Source):
    def __init__(self, wavelength, tol=1e-14, **kwargs):
        super().__init__(wavelength, dim=3, **kwargs)
        self.tol = tol
        r3D = "sqrt(pow(x[0],2) + pow(x[1],2) + pow(x[2],2))"
        r2D = "sqrt(pow(x[0],2) + pow(x[1],2))"
        self.r = df.Expression(
            f"{r3D}",
            **kwargs,
        )
        self.cos_theta = df.Expression(
            f"x[2]/({self.tol} + {r3D})",
            **kwargs,
        )
        self.sin_theta = df.Expression(
            f"{r2D}/({self.tol} + {r3D})",
            **kwargs,
        )
        self.cos_phi = df.Expression(
            f"x[0]/({self.tol} + {r2D})",
            **kwargs,
        )
        self.sin_phi = df.Expression(
            f"x[1]/({self.tol} + {r2D})",
            **kwargs,
        )
        ct = self.cos_theta
        st = self.sin_theta
        cp = self.cos_phi
        sp = self.sin_phi
        r = self.r
        rot_mat = [
            [st * cp, ct * cp, -sp],
            [st * sp, ct * sp, cp],
            [ct, -st, df.Constant(0)],
        ]
        self.transfo_matrix = Complex(df.as_tensor(rot_mat))

    def phasor(self, m):
        if m == 0:
            return Constant(1 + 0j)
        else:
            return Complex(self.cos_phi, self.sin_phi) ** df.Constant(m)


class SphericalHarmonic(SphericalSource):
    def __init__(self, wavelength, n, m, **kwargs):
        super().__init__(wavelength, **kwargs)
        self.n = n
        self.m = m
        self.phm = self.phasor(self.m)
        self.components = {}


class SphericalX(SphericalHarmonic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.components["r"] = Constant(0)
        self.components["theta"] = j * u(self.cos_theta, self.n, self.m) * self.phm
        self.components["phi"] = -s(self.cos_theta, self.n, self.m) * self.phm
        self.expression_spherical = vector(self.components.values())
        self.expression = self.transfo_matrix * self.expression_spherical


class SphericalY(SphericalHarmonic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.components["r"] = P(self.cos_theta, self.n, self.m) * self.phm
        self.components["theta"] = Constant(0)
        self.components["phi"] = Constant(0)
        self.expression_spherical = vector(self.components.values())
        self.expression = self.transfo_matrix * self.expression_spherical


class SphericalZ(SphericalHarmonic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.components["r"] = Constant(0)
        self.components["theta"] = s(self.cos_theta, self.n, self.m) * self.phm
        self.components["phi"] = j * u(self.cos_theta, self.n, self.m) * self.phm
        self.expression_spherical = vector(self.components.values())
        self.expression = self.transfo_matrix * self.expression_spherical


class SphericalM(SphericalHarmonic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        k = df.Constant(self.wavenumber)
        jn = sph_bessel_J(self.n, k * self.r)
        Xnm = SphericalX(*args, **kwargs)
        self.expression_spherical = jn * Xnm.expression_spherical
        self.expression = self.transfo_matrix * self.expression_spherical


class SphericalN(SphericalHarmonic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        k = df.Constant(self.wavenumber)
        jn = sph_bessel_J(self.n, k * self.r)
        psin_prime = ricatti_bessel_J(self.n - 1, k * self.r) - self.n * jn
        Y = SphericalY(*args, **kwargs)
        Z = SphericalZ(*args, **kwargs)
        Ynm = Y.expression_spherical
        Znm = Z.expression_spherical
        self.expression_spherical = (
            (self.n * (self.n + 1)) ** 0.5 * jn * Ynm + psin_prime * Znm
        ) / (k * self.r)
        self.expression = self.transfo_matrix * self.expression_spherical
