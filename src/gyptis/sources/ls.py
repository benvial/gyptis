#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

from .source import *


def green_function_2d(wavelength, xs, ys, phase=0, amplitude=1, degree=1, domain=None):
    """
    Compute the 2D Green function associated with a point source.

    Parameters
    ----------
    wavelength : float
        The wavelength of the Green function.
    xs : float
        The x-coordinate of the point source.
    ys : float
        The y-coordinate of the point source.
    phase : float, optional
        The phase shift of the Green function. Default is 0.
    amplitude : float, optional
        The amplitude of the Green function. Default is 1.
    degree : int, optional
        The degree of the output Expression. Default is 1.
    domain : dolfin.cpp.mesh.Mesh, optional
        The mesh for the domain of definition of the function.

    Returns
    -------
    expr : Expression
        The Green function as a dolfin Expression.
    """

    Xs = sympyvector(sp.symbols("xs, ys, 0", real=True))
    k0 = sp.symbols("k0", real=True)
    Xshift = X - Xs
    rho = sp.sqrt(Xshift.dot(Xshift))
    rho = rho.subs(x[2], 0)
    kr = k0 * rho
    k0_ = 2 * np.pi / wavelength
    KR = dolfin.Expression(
        sp.printing.ccode(kr), k0=k0_, xs=xs, ys=ys, degree=degree, domain=domain
    )
    re = dolfin.Expression("y0(KR)", KR=KR, degree=degree, domain=domain)
    im = dolfin.Expression("j0(KR)", KR=KR, degree=degree, domain=domain)
    return (
        -1
        / 4
        * Complex(re, im)
        * Constant(amplitude)
        * phase_shift_constant(ConstantRe(phase))
    )


class LineSource(Source):
    """
    LineSource class.

    Parameters
    ----------
    wavelength : float
        The wavelength of the line source.
    position : tuple of float
        The (x, y) coordinates of the line source position.
    dim : int, optional
        The dimension of the line source. Default is 2.
    phase : float, optional
        The phase shift of the line source. Default is 0.
    amplitude : float, optional
        The amplitude of the line source. Default is 1.
    degree : int, optional
        The degree of the output Expression. Default is 1.
    domain : dolfin.cpp.mesh.Mesh, optional
        The mesh for the domain of definition of the function.

    Raises
    ------
    NotImplementedError
        If the dimension `dim` is 3, since LineSource is not implemented in 3D.
    """

    def __init__(
        self, wavelength, position, dim=2, phase=0, amplitude=1, degree=1, domain=None
    ):
        if dim == 3:
            raise NotImplementedError("LineSource not implemented in 3D")
        super().__init__(
            wavelength,
            dim=dim,
            phase=phase,
            amplitude=amplitude,
            degree=degree,
            domain=domain,
        )
        self.position = position

    @property
    def expression(self):
        """
        The expression of the line source.

        Returns
        -------
        expr : Expression
            The expression of the line source as a dolfin Expression.
        """
        return green_function_2d(
            self.wavelength,
            *self.position,
            phase=self.phase,
            amplitude=self.amplitude,
            degree=self.degree,
            domain=self.domain,
        )
