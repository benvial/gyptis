#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


from .maxwell3d import *


class Maxwell3DBands(Maxwell3D):
    def __init__(self, *args, propagation_vector=(0, 0, 0), **kwargs):
        super().__init__(*args, **kwargs, modal=True)
        self.propagation_vector = propagation_vector

    @property
    def phasor_vect(self):

        return [
            phasor(
                self.propagation_vector[i],
                direction=i,
                degree=self.degree,
                domain=self.geometry.mesh,
            )
            for i in range(3)
        ]

    @property
    def phasor(self):
        return self.phasor_vect[0] * self.phasor_vect[1] * self.phasor_vect[2]

    @property
    def weak(self):
        u = self.trial * self.phasor
        v = self.test * self.phasor.conj
        return super()._weak(u, v, Constant((0, 0, 0)))
