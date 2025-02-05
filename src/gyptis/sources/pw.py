#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


from .source import *


def plane_wave_2d(wavelength, theta, phase=0, amplitude=1, degree=1, domain=None):
    """
    Compute a 2D plane wave.

    Parameters
    ----------
    wavelength : float
        The wavelength of the plane wave.
    theta : float
        The angle of incidence in radians.
    phase : float, optional
        The phase shift of the plane wave. Default is 0.
    amplitude : float, optional
        The amplitude of the plane wave. Default is 1.
    degree : int, optional
        The degree of the output Expression. Default is 1.
    domain : dolfin.cpp.mesh.Mesh, optional
        The mesh for the domain of definition of the function.

    Returns
    -------
    expr : Expression
        The 2D plane wave as a dolfin Expression.
    """

    k0 = 2 * np.pi / wavelength
    K = k0 * np.array((-np.sin(theta), -np.cos(theta)))
    K_ = sympyvector(sp.symbols("kx, ky, 0", real=True))
    expr = amplitude * sp.exp(1j * (K_.dot(X) + phase))
    return expression2complex_2d(expr, kx=K[0], ky=K[1], degree=degree, domain=domain)


def plane_wave_3d(
    wavelength, theta, phi, psi, phase=(0, 0, 0), amplitude=1, degree=1, domain=None
):
    """
    Compute a 3D plane wave.

    Parameters
    ----------
    wavelength : float
        The wavelength of the plane wave.
    theta : float
        The angle of incidence in radians.
    phi : float
        The azimuthal angle of incidence in radians.
    psi : float
        The polarization angle of incidence in radians.
    phase : tuple of float, optional
        The phase shifts of the x, y, and z components of the plane wave.
        Default is (0, 0, 0).
    amplitude : float, optional
        The amplitude of the plane wave. Default is 1.
    degree : int, optional
        The degree of the output Expression. Default is 1.
    domain : dolfin.cpp.mesh.Mesh, optional
        The mesh for the domain of definition of the function.

    Returns
    -------
    expr : Expression
        The 3D plane wave as a dolfin Expression.
    """
    cx = np.cos(psi) * np.cos(theta) * np.cos(phi) - np.sin(psi) * np.sin(phi)
    cy = np.cos(psi) * np.cos(theta) * np.sin(phi) + np.sin(psi) * np.cos(phi)
    cz = -np.cos(psi) * np.sin(theta)

    c = np.array([cx, cy, cz])
    phase_shifts = np.array([np.exp((phis)) for phis in phase])
    Carray = c * phase_shifts
    C = dolfin.as_tensor([df.Constant(_c) for _c in Carray])

    k0 = 2 * np.pi / wavelength
    K = k0 * np.array(
        (
            -np.sin(theta) * np.cos(phi),
            -np.sin(theta) * np.sin(phi),
            -np.cos(theta),
        )
    )
    K_ = sympyvector(sp.symbols("kx, ky, kz", real=True))

    Propp = amplitude * sp.exp(1j * (K_.dot(X)))

    code = [sp.printing.ccode(p) for p in Propp.as_real_imag()]
    prop = dolfin.Expression(
        code, kx=K[0], ky=K[1], kz=K[2], degree=degree, domain=domain
    )

    return Complex(prop[0] * C, prop[1] * C)


class PlaneWave(Source):
    """
    PlaneWave class.

    Parameters
    ----------
    wavelength : float
        The wavelength of the plane wave.
    angle : float
        The angle of incidence in radians.
    dim : int, optional
        The dimension of the plane wave. Default is 2.
    phase : float or list of float, optional
        The phase shift of the plane wave. Default is 0.
        If dim is 3, phase can be a list of three floats.
    amplitude : float, optional
        The amplitude of the plane wave. Default is 1.
    degree : int, optional
        The degree of the output Expression. Default is 1.
    domain : dolfin.cpp.mesh.Mesh, optional
        The mesh for the domain of definition of the function.

    Returns
    -------
    expr : Expression
        The plane wave as a dolfin Expression.
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
        """
        The plane wave as a dolfin Expression.

        Returns
        -------
        expr : Expression
            The plane wave as a dolfin Expression.
        """

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
