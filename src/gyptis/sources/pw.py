#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


from .source import *


def plane_wave_2d(wavelength, theta, phase=0, amplitude=1, degree=1, domain=None):
    k0 = 2 * np.pi / wavelength
    K = k0 * np.array((-np.sin(theta), -np.cos(theta)))
    K_ = vector(sp.symbols("kx, ky, 0", real=True))
    expr = amplitude * sp.exp(1j * (K_.dot(X) + phase))
    return expression2complex_2d(expr, kx=K[0], ky=K[1], degree=degree, domain=domain)


def plane_wave_3d(
    wavelength, theta, phi, psi, phase=(0, 0, 0), amplitude=1, degree=1, domain=None
):
    cx = np.cos(psi) * np.cos(theta) * np.cos(phi) - np.sin(psi) * np.sin(phi)
    cy = np.cos(psi) * np.cos(theta) * np.sin(phi) + np.sin(psi) * np.cos(phi)
    cz = -np.cos(psi) * np.sin(theta)

    k0 = 2 * np.pi / wavelength
    K = k0 * np.array(
        (
            -np.sin(theta) * np.cos(phi),
            -np.sin(theta) * np.sin(phi),
            -np.cos(theta),
        )
    )
    K_ = vector(sp.symbols("kx, ky, kz", real=True))

    phase_shifts = [np.exp((phis)) for phis in phase]
    Propp = amplitude * sp.exp(1j * (K_.dot(X)))

    code = [sp.printing.ccode(p) for p in Propp.as_real_imag()]
    prop = dolfin.Expression(
        code, kx=K[0], ky=K[1], kz=K[2], degree=degree, domain=domain
    )
    C = dolfin.as_tensor(
        [cx * phase_shifts[0], cy * phase_shifts[1], cz * phase_shifts[2]]
    )
    return Complex(prop[0] * C, prop[1] * C)


class PlaneWave(Source):
    """Short summary.

    Parameters
    ----------
    wavelength : float
        The wavelength of the plane wave.
    angle : float if `dim=2` tuple of len 3 if `dim=3`
        Angle(s) in radians.
    dim : int
        Dimension (2 or 3, the default is 2).
    phase : float
        Phase shift (in radian, the default is 0).
    amplitude : float
        Amplitude (the default is 1).
    degree : int
        Degree of the functional space (default is 1).
    domain : dolfin.cpp.mesh.Mesh
        The mesh for the domain of definition of the function.

    Attributes
    ----------
    angle

    """

    def __init__(
        self, wavelength, angle, dim=2, phase=0, amplitude=1, degree=1, domain=None
    ):
        if dim == 3 and np.isscalar(phase):
            phase = (phase, phase, phase)
        super().__init__(
            wavelength,
            dim,
            phase=phase,
            amplitude=amplitude,
            degree=degree,
            domain=domain,
        )
        self.angle = angle

    @property
    def expression(self):
        return (
            plane_wave_2d(
                self.wavelength,
                self.angle,
                phase=self.phase,
                amplitude=self.amplitude,
                degree=self.degree,
                domain=self.domain,
            )
            if self.dim == 2
            else plane_wave_3d(
                self.wavelength,
                *self.angle,
                phase=self.phase,
                amplitude=self.amplitude,
                degree=self.degree,
                domain=self.domain,
            )
        )
