#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


from .source import *


def plane_wave_2d(wavelength, theta, amplitude=1, degree=1, domain=None):
    k0 = 2 * np.pi / wavelength
    K = k0 * np.array((-np.sin(theta), -np.cos(theta)))
    K_ = vector(sp.symbols("kx, ky, 0", real=True))
    expr = amplitude * sp.exp(1j * K_.dot(X))
    return expression2complex_2d(expr, kx=K[0], ky=K[1], degree=degree, domain=domain)


def plane_wave_3d(wavelength, theta, phi, psi, amplitude=1, degree=1, domain=None):

    cx = np.cos(psi) * np.cos(theta) * np.cos(phi) - np.sin(psi) * np.sin(phi)
    cy = np.cos(psi) * np.cos(theta) * np.sin(phi) + np.sin(psi) * np.cos(phi)
    cz = -np.cos(psi) * np.sin(theta)

    k0 = 2 * np.pi / wavelength
    K = k0 * np.array(
        (
            -np.sin(theta) * np.cos(phi),
            -np.sin(theta) * np.sin(phi),
            -np.cos(theta),
        )
    )
    K_ = vector(sp.symbols("kx, ky, kz", real=True))

    Propp = amplitude * sp.exp(1j * K_.dot(X))

    code = [sp.printing.ccode(p) for p in Propp.as_real_imag()]
    prop = dolfin.Expression(
        code, kx=K[0], ky=K[1], kz=K[2], degree=degree, domain=domain
    )
    C = dolfin.as_tensor([cx, cy, cz])
    return Complex(prop[0] * C, prop[1] * C)


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
