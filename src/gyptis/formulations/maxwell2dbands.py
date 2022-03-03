#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


from .maxwell2d import *


class Maxwell2DBands(Maxwell2D):
    def __init__(self, *args, propagation_vector=(0, 0), **kwargs):
        super().__init__(*args, **kwargs, modal=True)
        self.propagation_vector = propagation_vector

    @property
    def phasor(self):
        _phasor = phasor(
            self.propagation_vector[0],
            direction=0,
            degree=self.degree,
            domain=self.geometry.mesh,
        )
        _phasor *= phasor(
            self.propagation_vector[1],
            direction=1,
            degree=self.degree,
            domain=self.geometry.mesh,
        )
        return _phasor

    @property
    def weak(self):
        u = self.trial * self.phasor
        v = self.test * self.phasor.conj
        return super()._weak(u, v, Constant(0))
