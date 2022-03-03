#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


from .source import *


class GaussianBeam(Source):
    def __init__(
        self, wavelength, angle, waist, dim, Npw=101, amplitude=1, degree=1, domain=None
    ):
        super().__init__(wavelength, dim, degree=degree, domain=domain)
        self.angle = angle
        self.amplitude = amplitude
        self.Npw = Npw
        self.waist = waist

    @property
    def expression(self):
        if self.dim == 2:
            _expression = Constant(0)
            for t in np.linspace(-np.pi / 2, np.pi / 2, self.Npw):
                _expression += (
                    plane_wave_2d(
                        self.wavelength,
                        self.angle + t,
                        amplitude=self.amplitude,
                        degree=self.degree,
                        domain=self.domain,
                    )
                    * Constant(
                        np.exp(-(t ** 2) * 4 * (np.pi / 2) ** 2 * self.waist ** 2)
                    )
                )
            dk = np.pi / (self.Npw - 1)
            _expression *= Constant(dk)
        else:
            raise (NotImplementedError)
        return _expression
