#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

"""
Sources.
"""

import numpy as np
import sympy as sp
from scipy.constants import c, epsilon_0, mu_0
from sympy.vector import CoordSys3D

from .complex import Complex, Constant, as_tensor, dolfin
from .complex import grad as gradc

_COORD = CoordSys3D("N")


def _vector(components):
    return (
        components[0] * _COORD.i + components[1] * _COORD.j + components[2] * _COORD.k
    )


_x = sp.symbols("x[0] x[1] x[2]", real=True)
_X = _vector(np.array(_x))


def _expression2complex_2d(expr, **kwargs):
    re, im = (p.subs(_x[2], 0) for p in expr.as_real_imag())
    dexpr = [dolfin.Expression(sp.printing.ccode(p), **kwargs) for p in (re, im)]
    return Complex(*dexpr)


def plane_wave_2d(lambda0, theta, amplitude=1, degree=1, domain=None):
    k0 = 2 * np.pi / lambda0
    K = k0 * np.array((np.cos(theta), np.sin(theta)))
    K_ = _vector(sp.symbols("kx, ky, 0", real=True))
    expr = amplitude * sp.exp(1j * K_.dot(_X))
    return _expression2complex_2d(expr, kx=K[0], ky=K[1], degree=degree, domain=domain)


def plane_wave_3d(lambda0, theta, phi, psi, amplitude=1, degree=1, domain=None):

    cx = np.cos(psi) * np.cos(theta) * np.cos(phi) - np.sin(psi) * np.sin(phi)
    cy = np.cos(psi) * np.cos(theta) * np.sin(phi) + np.sin(psi) * np.cos(phi)
    cz = -np.cos(psi) * np.sin(theta)

    k0 = 2 * np.pi / lambda0
    K = k0 * np.array(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        )
    )
    K_ = _vector(sp.symbols("kx, ky, kz", real=True))

    Propp = amplitude * sp.exp(1j * K_.dot(_X))

    code = [sp.printing.ccode(p) for p in Propp.as_real_imag()]
    prop = dolfin.Expression(
        code, kx=K[0], ky=K[1], kz=K[2], degree=degree, domain=domain
    )
    C = dolfin.as_tensor([cx, cy, cz])
    return Complex(prop[0] * C, prop[1] * C)


def green_function_2d(lambda0, xs, ys, amplitude=1, degree=1, domain=None):
    Xs = _vector(sp.symbols("xs, ys, 0", real=True))
    k0 = sp.symbols("k0", real=True)
    Xshift = _X - Xs
    rho = sp.sqrt(Xshift.dot(Xshift))
    rho = rho.subs(_x[2], 0)
    kr = k0 * rho
    k0_ = 2 * np.pi / lambda0
    KR = dolfin.Expression(
        sp.printing.ccode(kr), k0=k0_, xs=xs, ys=ys, degree=degree, domain=domain
    )
    return (
        -1
        / 4
        * Complex(dolfin.bessel_Y(0, KR), dolfin.bessel_J(0, KR))
        * Constant(amplitude)
    )


def field_stack_2D(phi, alpha, beta, yshift=0, degree=1, domain=None):
    alpha0_re, alpha0_im, beta0_re, beta0_im = sp.symbols(
        "alpha0_re,alpha0_im,beta0_re,beta0_im", real=True
    )
    alpha0 = alpha0_re + 1j * alpha0_im
    beta0 = beta0_re + 1j * beta0_im
    Kplus = _vector((alpha0, beta0, 0))
    Kminus = _vector((alpha0, -beta0, 0))
    deltaY = _vector((0, sp.symbols("yshift", real=True), 0))
    pw = lambda K: sp.exp(1j * K.dot(_X - deltaY))
    phi_plus_re, phi_plus_im = sp.symbols("phi_plus_re,phi_plus_im", real=True)
    phi_minus_re, phi_minus_im = sp.symbols("phi_minus_re,phi_minus_im", real=True)
    phi_plus = phi_plus_re + 1j * phi_plus_im
    phi_minus = phi_minus_re + 1j * phi_minus_im
    field = phi_plus * pw(Kplus) + phi_minus * pw(Kminus)

    expr = _expression2complex_2d(
        field,
        alpha0_re=alpha.real,
        alpha0_im=alpha.imag,
        beta0_re=beta.real,
        beta0_im=beta.imag,
        phi_plus_re=phi[0].real,
        phi_plus_im=phi[0].imag,
        phi_minus_re=phi[1].real,
        phi_minus_im=phi[1].imag,
        yshift=yshift,
        degree=degree,
        domain=domain,
    )
    return expr


def field_stack_3D(phi, alpha, beta, gamma, zshift=0, degree=1, domain=None):
    alpha0_re, alpha0_im, beta0_re, beta0_im, gamma0_re, gamma0_im = sp.symbols(
        "alpha0_re, alpha0_im, beta0_re, beta0_im, gamma0_re, gamma0_im", real=True
    )
    alpha0 = alpha0_re + 1j * alpha0_im
    beta0 = beta0_re + 1j * beta0_im
    gamma0 = gamma0_re + 1j * gamma0_im
    Kplus = _vector((alpha0, beta0, gamma0))
    Kminus = _vector((alpha0, beta0, -gamma0))
    deltaZ = _vector((0, 0, sp.symbols("zshift", real=True)))
    pw = lambda K: sp.exp(1j * K.dot(_X - deltaZ))
    fields = []
    for comp in ["x", "y", "z"]:
        phi_plus_re, phi_plus_im = sp.symbols(
            f"phi_plus_{comp}_re,phi_plus_{comp}_im", real=True
        )
        phi_minus_re, phi_minus_im = sp.symbols(
            f"phi_minus_{comp}_re,phi_minus_{comp}_im", real=True
        )
        phi_plus = phi_plus_re + 1j * phi_plus_im
        phi_minus = phi_minus_re + 1j * phi_minus_im
        field = phi_plus * pw(Kplus) + phi_minus * pw(Kminus)
        fields.append(field)
    code = [[sp.printing.ccode(p) for p in f.as_real_imag()] for f in fields]
    code = np.ravel(code).tolist()
    expr = [
        dolfin.Expression(
            c,
            alpha0_re=alpha.real,
            alpha0_im=alpha.imag,
            beta0_re=beta.real,
            beta0_im=beta.imag,
            gamma0_re=gamma.real,
            gamma0_im=gamma.imag,
            phi_plus_x_re=phi[0].real,
            phi_plus_x_im=phi[0].imag,
            phi_minus_x_re=phi[1].real,
            phi_minus_x_im=phi[1].imag,
            phi_plus_y_re=phi[2].real,
            phi_plus_y_im=phi[2].imag,
            phi_minus_y_re=phi[3].real,
            phi_minus_y_im=phi[3].imag,
            phi_plus_z_re=phi[4].real,
            phi_plus_z_im=phi[4].imag,
            phi_minus_z_re=phi[5].real,
            phi_minus_z_im=phi[5].imag,
            zshift=zshift,
            degree=degree,
            domain=domain,
        )
        for c in code
    ]

    return Complex(
        as_tensor([expr[0], expr[2], expr[4]]), as_tensor([expr[1], expr[3], expr[5]])
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


class PlaneWave(Source):
    def __init__(self, wavelength, angle, dim, amplitude=1, degree=1, domain=None):
        super().__init__(wavelength, dim, degree=degree, domain=domain)
        self.angle = angle
        self.amplitude = amplitude

    @property
    def expression(self):
        if self.dim == 2:
            _expression = plane_wave_2d(
                self.wavelength,
                self.angle,
                amplitude=self.amplitude,
                degree=self.degree,
                domain=self.domain,
            )
        else:
            _expression = plane_wave_3d(
                self.wavelength,
                *self.angle,
                amplitude=self.amplitude,
                degree=self.degree,
                domain=self.domain,
            )
        return _expression


class LineSource(Source):
    def __init__(self, wavelength, position, amplitude=1, degree=1, domain=None):
        super().__init__(wavelength, dim=2, degree=degree, domain=domain)
        self.position = position
        self.amplitude = amplitude

    @property
    def expression(self):
        _expression = green_function_2d(
            self.wavelength,
            *self.position,
            amplitude=self.amplitude,
            degree=self.degree,
            domain=self.domain,
        )
        return _expression


#
# mesh = dolfin.UnitSquareMesh(50, 50)
# pw = PlaneWave(0.3, 0*np.pi/2, 2, degree=2)
#
# # pw = plane_wave_2d(0.1, 0, degree=2)
#
# from gyptis.plot import *
#
# plt.ion()
#
# W = dolfin.FunctionSpace(mesh,"CG",2)
#
# Wc = ComplexFunctionSpace(mesh,"CG",2)
#
# # pwexp = project(pw.expression.real, W)
# # pwexp = project(pw.expression, W)
#
# plt.close("all")
# plt.figure()
# cb = dolfin.plot(pw.expression.real, mesh=mesh)
# cb = dolfin.plot(pw.gradient[0].real, mesh=mesh)
# plt.colorbar(cb)
#
#
# plt.figure()
# pw.wavelength=0.1
# pw.angle=0.2
# cb = dolfin.plot(pw.expression.real, mesh=mesh)
# cb = dolfin.plot(pw.gradient[0].real, mesh=mesh)
# plt.colorbar(cb)
#
# #
# # cdsc
# # plt.clf()
# #
# # cb = dolfin.plot(pw.gradient[0].real, mesh=mesh)
# # plt.colorbar(cb)
# # # dolfin.plot(pw.gradient[0].imag, mesh=mesh)
