#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


from .maxwell2d import *


class Maxwell2DPeriodic(Maxwell2D):
    def __init__(self, *args, propagation_constant=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.propagation_constant = propagation_constant

        if self.modal:
            self.propagation_vector = np.array([self.propagation_constant, 0])

        else:
            self.propagation_vector = self.source.wavenumber * np.array(
                [-np.sin(self.source.angle), -np.cos(self.source.angle)]
            )
        self.phasor = phasor(
            self.propagation_vector[0],
            direction=0,
            degree=self.degree,
            domain=self.geometry.mesh,
        )
        self.annex_field = (
            make_stack(
                self.geometry,
                self.coefficients,
                self.source,
                polarization=self.polarization,
                source_domains=self.source_domains,
                degree=self.degree,
                dim=2,
            )
            if not self.modal
            else None
        )

    @property
    def weak(self):
        u1 = self.annex_field["as_subdomain"]["stack"] if not self.modal else 0
        u = self.trial * self.phasor
        v = self.test * self.phasor.conj
        return super()._weak(u, v, u1)

    def build_boundary_conditions(self):

        applied_function = (
            Constant(0)
            if self.modal
            else -self.annex_field["as_subdomain"]["stack"] * self.phasor.conj
        )
        self._boundary_conditions = self.build_pec_boundary_conditions(applied_function)
        return self._boundary_conditions
