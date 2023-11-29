#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

from .ls import *


class Dipole(Source):
    def __init__(
        self,
        wavelength,
        position,
        angle=0,
        dim=2,
        phase=0,
        amplitude=1,
        degree=1,
        domain=None,
    ):
        if dim == 3:
            raise NotImplementedError("Dipole not implemented in 3D")
        super().__init__(
            wavelength,
            dim=dim,
            phase=phase,
            amplitude=amplitude,
            degree=degree,
            domain=domain,
        )
        self.position = position
        self.angle = angle

    @property
    def expression(self):
        ls = LineSource(
            self.wavelength,
            self.position,
            self.dim,
            self.phase,
            self.amplitude,
            self.degree + 1,
            self.domain,
        )
        n = as_vector(
            [ConstantRe(-np.sin(self.angle)), ConstantRe(-np.cos(self.angle))]
        )
        dls = grad(ls.expression)
        return dot(dls, n) / Constant(1j * self.wavenumber)
