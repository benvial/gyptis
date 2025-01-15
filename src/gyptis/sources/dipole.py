#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

from .ls import *


class Dipole(Source):
    """
    Dipole class.

    This class defines a dipole source for 2D electromagnetic simulations.

    Parameters
    ----------
    wavelength : float
        The wavelength of the dipole source.
    position : tuple of float
        The (x, y) coordinates of the dipole position.
    angle : float, optional
        The orientation angle of the dipole in radians. Default is 0.
    dim : int, optional
        The spatial dimension of the source. Default is 2.
    phase : float, optional
        The phase shift of the source. Default is 0.
    amplitude : float, optional
        The amplitude of the source. Default is 1.
    degree : int, optional
        The degree of the output Expression. Default is 1.
    domain : dolfin.cpp.mesh.Mesh, optional
        The mesh for the domain of definition of the function.

    Raises
    ------
    NotImplementedError
        If the dimension `dim` is 3, since Dipole is not implemented in 3D.
    """

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
        """
        The expression of the dipole.

        The expression is given by the gradient of a line source
        multiplied by a unit vector in the direction of the dipole.

        Returns
        -------
        expr : Expression
            The expression of the dipole as a dolfin Expression.
        """
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
