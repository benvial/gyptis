#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


from .pw import *


class GaussianBeam(Source):
    """
    GaussianBeam class.

    A Gaussian beam is a common type of beam in optics. It is a solution of the
    paraxial Helmholtz equation, which is a simplified form of the wave equation
    that is only valid for waves propagating in a single direction. The Gaussian
    beam is a good model for the output of a laser, or for a beam that has been
    focused by a lens.

    Parameters
    ----------
    wavelength : float
        The wavelength of the Gaussian beam.
    angle : float
        The angle of incidence of the Gaussian beam.
    waist : float
        The waist of the Gaussian beam, which is the minimum size of the beam.
    position : tuple, optional
        The position of the Gaussian beam. Default is (0, 0).
    dim : int, optional
        The dimension of the Gaussian beam. Default is 2.
    Npw : int, optional
        The number of plane waves used to approximate the Gaussian beam. Default
        is 101.
    phase : float or list of float, optional
        The phase shift of the Gaussian beam. Default is 0.
        If dim is 3, phase can be a list of three floats.
    amplitude : float, optional
        The amplitude of the Gaussian beam. Default is 1.
    degree : int, optional
        The degree of the output Expression. Default is 1.
    domain : dolfin.cpp.mesh.Mesh, optional
        The mesh for the domain of definition of the function.

    Returns
    -------
    expr : Expression
        The Gaussian beam as a dolfin Expression.
    """

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
        """
        The Gaussian beam as a dolfin Expression.

        The Gaussian beam is modelled as a sum of plane waves with a Gaussian
        distribution of angles.

        Returns
        -------
        expr : Expression
            The Gaussian beam as a dolfin Expression.
        """
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
                    -(t**2) * 4 * (np.pi / 2) ** 2 * (self.waist / self.wavelength) ** 2
                )
            )
            term *= phase_shift_constant(ConstantRe(phase_pos))
            _expression += term
        dk = np.pi / (self.Npw - 1)
        _expression *= Constant(dk)
        return _expression
