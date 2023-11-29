#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

from .source import *


def green_function_2d(wavelength, xs, ys, phase=0, amplitude=1, degree=1, domain=None):
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
        * phase_shift_constant(ConstantRe(phase))
    )


class LineSource(Source):
    def __init__(
        self, wavelength, position, dim=2, phase=0, amplitude=1, degree=1, domain=None
    ):
        if dim == 3:
            raise NotImplementedError("LineSource not implemented in 3D")
        super().__init__(
            wavelength,
            dim=dim,
            phase=phase,
            amplitude=amplitude,
            degree=degree,
            domain=domain,
        )
        self.position = position

    @property
    def expression(self):
        return green_function_2d(
            self.wavelength,
            *self.position,
            phase=self.phase,
            amplitude=self.amplitude,
            degree=self.degree,
            domain=self.domain,
        )
