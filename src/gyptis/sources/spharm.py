#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
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
    """
    Compute a safe square root of a complex number.

    This function computes the square root of a complex number while avoiding
    the branch cut of the complex square root. The output is a complex number
    with positive real part.

    Parameters
    ----------
    x : Complex
        The complex number to compute the square root of.

    Returns
    -------
    Complex
        The square root of x with positive real part.
    """
    x = Complex(x, 0)
    t = x.phase
    return df.sqrt(x.module) * Complex(df.cos(t / 2), df.sin(t / 2))


def sph_bessel_J(nu, f):
    """
    Compute the spherical Bessel function of the first kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the spherical Bessel function.
    f : float
        The argument of the spherical Bessel function.

    Returns
    -------
    Complex
        The spherical Bessel function of the first kind of order nu.
    """
    return df.sqrt(pi / (2 * f)) * df.bessel_J(nu + 0.5, f)


def sph_bessel_Y(nu, f):
    """
    Compute the spherical Bessel function of the second kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the spherical Bessel function.
    f : float
        The argument of the spherical Bessel function.

    Returns
    -------
    Complex
        The spherical Bessel function of the second kind of order nu.
    """
    return df.sqrt(pi / (2 * f)) * df.bessel_Y(nu + 0.5, f)


def sph_bessel_H1(nu, f):
    """
    Compute the spherical Hankel function of the first kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the spherical Hankel function.
    f : float
        The argument of the spherical Hankel function.

    Returns
    -------
    Complex
        The spherical Hankel function of the first kind of order nu.
    """
    return Complex(sph_bessel_J(nu, f), sph_bessel_Y(nu, f))


def sph_bessel_H2(nu, f):
    """
    Compute the spherical Hankel function of the second kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the spherical Hankel function.
    f : float
        The argument of the spherical Hankel function.

    Returns
    -------
    Complex
        The spherical Hankel function of the second kind of order nu.
    """
    return Complex(sph_bessel_J(nu, f), -sph_bessel_Y(nu, f))


def ricatti_bessel_J(nu, f):
    """
    Compute the Ricatti-Bessel function of the first kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the Ricatti-Bessel function.
    f : float
        The argument of the Ricatti-Bessel function.

    Returns
    -------
    Complex
        The Ricatti-Bessel function of the first kind of order nu.
    """

    return f * sph_bessel_J(nu, f)


def ricatti_bessel_Y(nu, f):
    """
    Compute the Ricatti-Bessel function of the second kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the Ricatti-Bessel function.
    f : float
        The argument of the Ricatti-Bessel function.

    Returns
    -------
    Complex
        The Ricatti-Bessel function of the second kind of order nu.
    """

    return f * sph_bessel_Y(nu, f)


def ricatti_bessel_H1(nu, f):
    """
    Compute the Ricatti-Bessel function of the first kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the Ricatti-Bessel function.
    f : float
        The argument of the Ricatti-Bessel function.

    Returns
    -------
    Complex
        The Ricatti-Bessel function of the first kind of order nu.
    """

    return Complex(ricatti_bessel_J(nu, f), ricatti_bessel_Y(nu, f))


def ricatti_bessel_H2(nu, f):
    """
    Compute the Ricatti-Bessel function of the second kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the Ricatti-Bessel function.
    f : float
        The argument of the Ricatti-Bessel function.

    Returns
    -------
    Complex
        The Ricatti-Bessel function of the second kind of order nu.
    """
    return Complex(ricatti_bessel_J(nu, f), -ricatti_bessel_Y(nu, f))


def sph_jn(nu, f):
    """
    Compute the spherical Bessel function of the first kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the spherical Bessel function.
    f : float
        The argument of the spherical Bessel function.

    Returns
    -------
    float
        The spherical Bessel function of the first kind of order nu.
    """
    return (pi / (2 * f)) ** 0.5 * sp.jv(nu + 0.5, f)


def sph_yn(nu, f):
    """
    Compute the spherical Bessel function of the second kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the spherical Bessel function.
    f : float
        The argument of the spherical Bessel function.

    Returns
    -------
    float
        The spherical Bessel function of the second kind of order nu.
    """

    return (pi / (2 * f)) ** 0.5 * sp.yv(nu + 0.5, f)


def sph_hn1(nu, f):
    """
    Compute the spherical Hankel function of the first kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the spherical Hankel function.
    f : float
        The argument of the spherical Hankel function.

    Returns
    -------
    Complex
        The spherical Hankel function of the first kind of order nu.
    """
    return sph_jn(nu, f) + 1j * sph_yn(nu, f)


def sph_hn2(nu, f):
    """
    Compute the spherical Hankel function of the second kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the spherical Hankel function.
    f : float
        The argument of the spherical Hankel function.

    Returns
    -------
    Complex
        The spherical Hankel function of the second kind of order nu.
    """
    return sph_jn(nu, f) - 1j * sph_yn(nu, f)


def rb_jn(nu, f):
    """
    Compute the Ricatti-Bessel function of the first kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the Ricatti-Bessel function.
    f : float
        The argument of the Ricatti-Bessel function.

    Returns
    -------
    Complex
        The Ricatti-Bessel function of the first kind of order nu.
    """
    return f * sph_jn(nu, f)


def rb_yn(nu, f):
    """
    Compute the Ricatti-Bessel function of the second kind of order nu.

    Parameters
    ----------
    nu : float
        The order of the Ricatti-Bessel function.
    f : float
        The argument of the Ricatti-Bessel function.

    Returns
    -------
    Complex
        The Ricatti-Bessel function of the second kind of order nu.
    """
    return f * sph_yn(nu, f)


def rb_hn1(nu, f):
    """
    Compute the Ricatti-Bessel function of the first kind of order nu,
    using the spherical Bessel functions of the first and second kinds.

    Parameters
    ----------
    nu : float
        The order of the Ricatti-Bessel function.
    f : float
        The argument of the Ricatti-Bessel function.

    Returns
    -------
    Complex
        The Ricatti-Bessel function of the first kind of order nu.
    """

    return rb_jn(nu, f) + 1j * rb_yn(nu, f)


def rb_hn2(nu, f):
    """
    Compute the Ricatti-Bessel function of the second kind of order nu,
    using the spherical Bessel functions of the first and second kinds.

    Parameters
    ----------
    nu : float
        The order of the Ricatti-Bessel function.
    f : float
        The argument of the Ricatti-Bessel function.

    Returns
    -------
    Complex
        The Ricatti-Bessel function of the second kind of order nu.
    """

    return rb_jn(nu, f) - 1j * rb_yn(nu, f)


def nm2p(n, m):
    """
    Convert a pair of spherical harmonic indices (n, m) to a scalar index p.

    Parameters
    ----------
    n : int
        The degree of the spherical harmonic.
    m : int
        The order of the spherical harmonic.

    Returns
    -------
    int
        The scalar index of the spherical harmonic.
    """
    return n * (n + 1) - m


def p2nm(p):
    """
    Convert a scalar index p of a spherical harmonic to a pair of spherical
    harmonic indices (n, m).

    Parameters
    ----------
    p : int
        The scalar index of the spherical harmonic.

    Returns
    -------
    tuple
        The pair of spherical harmonic indices (n, m).
    """
    n = int(p**0.5)
    m = n * (n + 1) - p
    return n, m


# Recurrence relations for scalar spherical harmonics


def P(x, n, m, y=None):
    """
    Compute the associated Legendre polynomial of degree n and order m,
    evaluated at x.

    Parameters
    ----------
    x : float
        The argument of the associated Legendre polynomial.
    n : int
        The degree of the associated Legendre polynomial.
    m : int
        The order of the associated Legendre polynomial.
    y : float, optional
        The value of sqrt(1 - x**2). If not provided, it is computed.

    Returns
    -------
    float
        The associated Legendre polynomial of degree n and order m, evaluated
        at x.
    """
    y = y or sqrt_safe(1 - x**2)
    if m < 0:
        return (-1) ** (-m) * P(x, n, -m)
    elif n == m == 0:
        return 1 / (4 * pi) ** 0.5
    elif n == m:
        return -(((2 * n + 1) / (2 * n)) ** 0.5) * y * P(x, n - 1, m - 1)
    elif m == (n - 1):
        return x * (2 * n + 1) ** 0.5 * P(x, n - 1, n - 1)
    else:
        return (((2 * n + 1) / (n**2 - m**2)) ** 0.5) * (
            (2 * n - 1) ** 0.5 * x * P(x, n - 1, m)
            - (((n - 1) ** 2 - m**2) / (2 * n - 3)) ** 0.5 * P(x, n - 2, m)
        )


def u(x, n, m, y=None):
    """
    Compute the derivative of the associated Legendre polynomial of degree n and
    order m, evaluated at x.

    Parameters
    ----------
    x : float
        The argument of the associated Legendre polynomial.
    n : int
        The degree of the associated Legendre polynomial.
    m : int
        The order of the associated Legendre polynomial.
    y : float, optional
        The value of sqrt(1 - x**2). If not provided, it is computed.

    Returns
    -------
    float
        The derivative of the associated Legendre polynomial of degree n and
        order m, evaluated at x.
    """
    y = y or sqrt_safe(1 - x**2)
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
        return ((2 * n + 1) * (n - 1) / ((n + 1) * (n**2 - m**2))) ** 0.5 * (
            x * (2 * n - 1) ** 0.5 * u(x, n - 1, m)
            - ((n - 2) * ((n - 1) ** 2 - m**2) / (n * (2 * n - 3))) ** 0.5
            * u(x, n - 2, m)
        )


def s(x, n, m, y=None):
    """
    Compute the derivative of the associated Legendre polynomial of degree n
    and order m, with respect to its argument x, evaluated at x.

    Parameters
    ----------
    x : float
        The argument of the associated Legendre polynomial.
    n : int
        The degree of the associated Legendre polynomial.
    m : int
        The order of the associated Legendre polynomial.
    y : float, optional
        The value of sqrt(1 - x**2). If not provided, it is computed.

    Returns
    -------
    float
        The derivative of the associated Legendre polynomial of degree n and
        order m, with respect to its argument x, evaluated at x.
    """

    y = y or sqrt_safe(1 - x**2)
    if m < 0:
        return (-1) ** (-m) * s(x, n, -m)
    elif n == m:
        return x * u(x, n, m)
    else:
        return 1 / (m + 1) * ((n + m + 1) * (n - m)) ** 0.5 * y * u(
            x, n, m + 1
        ) + x * u(x, n, m)


class SphericalSource(Source):
    """
    Spherical source.

    Parameters
    ----------
    wavelength : float
        The wavelength of the source.
    tol : float, optional
        The tolerance for the spherical Bessel function evaluation.
    **kwargs : dict
        Optional keyword arguments to be passed to the `Source` class.

    Attributes
    ----------
    tol : float
        The tolerance for the spherical Bessel function evaluation.
    r : dolfin.Expression
        The radial expression.
    cos_theta : dolfin.Expression
        The cosinus of the polar angle expression.
    sin_theta : dolfin.Expression
        The sinus of the polar angle expression.
    cos_phi : dolfin.Expression
        The cosinus of the azimuthal angle expression.
    sin_phi : dolfin.Expression
        The sinus of the azimuthal angle expression.
    """

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
    """
    Spherical harmonic source class.

    This class defines a spherical harmonic source with a given degree `n` and order `m`.

    Parameters
    ----------
    wavelength : float
        The wavelength of the source.
    n : int
        The degree of the spherical harmonic.
    m : int
        The order of the spherical harmonic.
    **kwargs : dict
        Optional keyword arguments to be passed to the `Source` class.

    Attributes
    ----------
    n : int
        The degree of the spherical harmonic.
    m : int
        The order of the spherical harmonic.
    phm : dolfin.Expression
        The phasor of the spherical harmonic.
    components : dict
        The spherical components of the source.
    """

    def __init__(self, wavelength, n, m, **kwargs):
        super().__init__(wavelength, **kwargs)
        self.n = n
        self.m = m
        self.phm = self.phasor(self.m)
        self.components = {}

    @property
    def expression(self):
        return self.transfo_matrix * self.expression_spherical


class SphericalX(SphericalHarmonic):
    """
    SphericalX source class.

    Parameters
    ----------
    *args : tuple
        Positional arguments to be passed to the `SphericalHarmonic` class.
    **kwargs : dict
        Optional keyword arguments to be passed to the `SphericalHarmonic` class.

    Notes
    -----
    The spherical components of the source are defined as follows:

    - `r` component: 0
    - `theta` component: -j * u(cos(theta), n, m) * phm
    - `phi` component: s(cos(theta), n, m) * phm

    where `j` is the imaginary unit, `u` is the associated Legendre
    function, `s` is the derivative of the associated Legendre function with
    respect to the argument, and `phm` is the phasor of the spherical
    harmonic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.components["r"] = Constant(0)
        self.components["theta"] = j * u(self.cos_theta, self.n, self.m) * self.phm
        self.components["phi"] = -s(self.cos_theta, self.n, self.m) * self.phm
        self.expression_spherical = vector(self.components.values())


class SphericalY(SphericalHarmonic):
    """
    SphericalY source class.

    Parameters
    ----------
    *args : tuple
        Positional arguments to be passed to the `SphericalHarmonic` class.
    **kwargs : dict
        Optional keyword arguments to be passed to the `SphericalHarmonic` class.

    Notes
    -----
    The spherical components of the source are defined as follows:

    - `r` component: P(cos(theta), n, m) * phm
    - `theta` component: 0
    - `phi` component: 0

    where `P` is the associated Legendre function and `phm` is the phasor of the spherical
    harmonic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.components["r"] = P(self.cos_theta, self.n, self.m) * self.phm
        self.components["theta"] = Constant(0)
        self.components["phi"] = Constant(0)
        self.expression_spherical = vector(self.components.values())


class SphericalZ(SphericalHarmonic):
    """
    SphericalZ source class.

    Parameters
    ----------
    *args : tuple
        Positional arguments to be passed to the `SphericalHarmonic` class.
    **kwargs : dict
        Optional keyword arguments to be passed to the `SphericalHarmonic` class.

    Notes
    -----
    The spherical components of the source are defined as follows:

    - `r` component: 0
    - `theta` component: s(cos(theta), n, m) * phm
    - `phi` component: j * u(cos(theta), n, m) * phm

    where `j` is the imaginary unit, `u` is the associated Legendre
    function, `s` is the derivative of the associated Legendre function with
    respect to the argument, and `phm` is the phasor of the spherical
    harmonic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.components["r"] = Constant(0)
        self.components["theta"] = s(self.cos_theta, self.n, self.m) * self.phm
        self.components["phi"] = j * u(self.cos_theta, self.n, self.m) * self.phm
        self.expression_spherical = vector(self.components.values())


class SphericalM(SphericalHarmonic):
    """
    SphericalM source class.

    Parameters
    ----------
    *args : tuple
        Positional arguments to be passed to the `SphericalHarmonic` class.
    **kwargs : dict
        Optional keyword arguments to be passed to the `SphericalHarmonic` class.

    Notes
    -----
    The spherical components of the source are defined as follows:

    - `r` component: 0
    - `theta` component: j * jn(n, kr) * Xnm
    - `phi` component: 0

    where `j` is the imaginary unit, `jn` is the spherical Bessel function of
    degree `n`, `kr` is the product of the wave number `k` and the radial
    distance `r`, and `Xnm` is the SphericalX source of degree `n` and order `m`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        k = df.Constant(self.wavenumber)
        jn = sph_bessel_J(self.n, k * self.r)
        Xnm = SphericalX(*args, **kwargs)
        self.expression_spherical = jn * Xnm.expression_spherical


class SphericalN(SphericalHarmonic):
    """
    SphericalN source class.

    Parameters
    ----------
    *args : tuple
        Positional arguments to be passed to the `SphericalHarmonic` class.
    **kwargs : dict
        Optional keyword arguments to be passed to the `SphericalHarmonic` class.

    Notes
    -----
    The spherical components of the source are defined as a combination of
    the spherical Bessel function and the spherical harmonics:

    - `jn` is the spherical Bessel function of degree `n`.
    - `psin_prime` is the derivative of the Ricatti-Bessel function.
    - `Ynm` is the SphericalY source expression.
    - `Znm` is the SphericalZ source expression.

    These components are used to compute the expression for the SphericalN
    source.
    """

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
