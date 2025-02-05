#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

from .ls import *


def grad_green_function_2d(
    wavelength, xs, ys, phase=0, amplitude=1, degree=1, domain=None
):
    """
    Compute the gradient of the 2D Green function associated with a point source.

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
        The gradient of the Green function as a dolfin Expression.
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
    R = dolfin.Expression(
        sp.printing.ccode(rho), xs=xs, ys=ys, degree=degree, domain=domain
    )
    Xshift_list = list(Xshift.components.values())
    A = (
        1
        / 4
        * Complex(dolfin.bessel_Y(1, KR), dolfin.bessel_J(1, KR))
        * Constant(amplitude * k0_)
        / R
        * phase_shift_constant(ConstantRe(phase))
    )
    dg = [
        A
        * dolfin.Expression(
            sp.printing.ccode(coord), xs=xs, ys=ys, degree=degree, domain=domain
        )
        for coord in Xshift_list[:2]
    ]
    re = as_vector([_f.real for _f in dg])
    im = as_vector([_f.imag for _f in dg])
    out = Complex(re, im)
    return out


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
        dls = grad_green_function_2d(
            self.wavelength,
            self.position[0],
            self.position[1],
            self.phase,
            self.amplitude,
            self.degree,
            self.domain,
        )
        n = as_vector(
            [ConstantRe(-np.sin(self.angle)), ConstantRe(-np.cos(self.angle))]
        )

        ### old lazy implementation
        # ls = LineSource(
        #     self.wavelength,
        #     self.position,
        #     self.dim,
        #     self.phase,
        #     self.amplitude,
        #     self.degree + 1,
        #     self.domain,
        # )
        # dls = grad(ls.expression)
        return dot(dls, n) / Constant(1j * self.wavenumber)
