#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


from .pw import *


class GaussianBeam(Source):
    def __init__(
        self,
        wavelength,
        angle,
        waist,
        position=(0, 0),
        dim=2,
        Npw=101,
        phase=0,
        amplitude=1,
        degree=1,
        domain=None,
    ):
        if dim == 3:
            raise NotImplementedError("GaussianBeam not implemented in 3D")
        super().__init__(
            wavelength,
            dim=dim,
            phase=phase,
            amplitude=amplitude,
            degree=degree,
            domain=domain,
        )
        self.angle = angle
        self.waist = waist
        self.position = position
        self.Npw = Npw

    @property
    def expression(self):
        _expression = Constant(0)
        for t in np.linspace(-np.pi / 2, np.pi / 2, self.Npw):
            angle_i = self.angle + t
            K = self.wavenumber * np.array((-np.sin(angle_i), -np.cos(angle_i)))
            position = np.array(self.position)
            phase_pos = -np.dot(K, position)
            term = plane_wave_2d(
                self.wavelength,
                angle_i,
                phase=self.phase,
                amplitude=self.amplitude,
                degree=self.degree,
                domain=self.domain,
            ) * Constant(
                np.exp(
                    -(t**2)
                    * 4
                    * (np.pi / 2) ** 2
                    * (self.waist / self.wavelength) ** 2
                )
            )
            term *= phase_shift_constant(ConstantRe(phase_pos))
            _expression += term
        dk = np.pi / (self.Npw - 1)
        _expression *= Constant(dk)
        return _expression
