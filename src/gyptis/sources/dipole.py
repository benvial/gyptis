#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

from .source import *


class Dipole(Source):
    def __init__(
        self, wavelength, position, angle=0, amplitude=1, degree=1, domain=None
    ):
        super().__init__(wavelength, dim=2, degree=degree, domain=domain)
        self.position = position
        self.amplitude = amplitude
        self.angle = angle

    @property
    def expression(self):
        ls = LineSource(
            self.wavelength, self.position, self.amplitude, self.degree + 1, self.domain
        )
        n = as_vector([ConstantRe(-np.sin(self.angle)), ConstantRe(np.cos(self.angle))])
        dls = grad(ls.expression)
        _expression = dot(dls, n) / Constant(1j / self.wavenumber)
        return _expression
