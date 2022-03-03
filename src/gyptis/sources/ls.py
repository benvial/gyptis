#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

from .source import *


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
