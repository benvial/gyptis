#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


from .maxwell3d import *


class Maxwell3DPeriodic(Maxwell3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        k0 = self.source.wavenumber
        theta0, phi0 = self.source.angle[0:2]
        alpha0 = -k0 * np.sin(theta0) * np.cos(phi0)
        beta0 = -k0 * np.sin(theta0) * np.sin(phi0)
        gamma0 = -k0 * np.cos(theta0)
        self.propagation_vector = np.array([alpha0, beta0, gamma0])

        self.phasor_vect = [
            phasor(
                self.propagation_vector[i],
                direction=i,
                degree=self.degree,
                domain=self.geometry.mesh,
            )
            for i in range(3)
        ]
        self.phasor = self.phasor_vect[0] * self.phasor_vect[1]
        self.annex_field = make_stack(
            self.geometry,
            self.coefficients,
            self.source,
            source_domains=self.source_domains,
            degree=self.degree,
            dim=3,
        )

    @property
    def weak(self):
        u1 = self.annex_field["as_subdomain"]["stack"]
        u = self.trial * self.phasor
        v = self.test * self.phasor.conj
        return self._weak(u, v, u1)

    def build_boundary_conditions(self):
        applied_function = -self.annex_field["as_subdomain"]["stack"] * self.phasor.conj
        self._boundary_conditions = self.build_pec_boundary_conditions(applied_function)
        return self._boundary_conditions
