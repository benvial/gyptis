#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

"""
Sources.
"""

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp
from dolfin import Constant as ConstantRe
from scipy.constants import c, epsilon_0, mu_0
from sympy.vector import CoordSys3D

from .. import dolfin
from ..complex import Complex, Constant, as_tensor, as_vector, dot, grad
from ..plot import *

_COORD = CoordSys3D("N")


def sympyvector(components):
    """
    Convert a 3D vector to a sympy.vector expression.

    Parameters
    ----------
    components : array-like of shape (3,)
        The components of the vector.

    Returns
    -------
    vector : sympy.vector.Vector
        The sympy vector.
    """
    return (
        components[0] * _COORD.i + components[1] * _COORD.j + components[2] * _COORD.k
    )


x = sp.symbols("x[0] x[1] x[2]", real=True)
X = sympyvector(np.array(x))


def expression2complex_2d(expr, **kwargs):
    """
    Convert a 2D expression to a complex dolfin expression.

    Parameters
    ----------
    expr : sympy expression
        The 2D expression to convert.
    **kwargs : dict
        Keyword arguments to be passed to dolfin.Expression.

    Returns
    -------
    Complex
        The complex dolfin expression.
    """
    re, im = (p.subs(x[2], 0) for p in expr.as_real_imag())
    dexpr = [dolfin.Expression(sp.printing.ccode(p), **kwargs) for p in (re, im)]
    return Complex(*dexpr)


class Source(ABC):
    """
    Abstract base class for defining a source in the gyptis framework.

    Parameters
    ----------
    wavelength : float
        The wavelength of the source.
    dim : int
        The dimension of the source.
    phase : float, optional
        The phase shift of the source. Default is 0.
    amplitude : float, optional
        The amplitude of the source. Default is 1.
    degree : int, optional
        The degree of the output Expression. Default is 1.
    domain : dolfin.cpp.mesh.Mesh, optional
        The mesh for the domain of definition of the function.

    """

    def __init__(self, wavelength, dim, phase=0, amplitude=1, degree=1, domain=None):
        self.wavelength = wavelength
        self.phase = phase
        self.amplitude = amplitude
        self.dim = dim
        self.degree = degree
        self.domain = domain

    @property
    def wavenumber(self):
        """
        Calculate the wave number from the wavelength.

        Returns
        -------
        float
            The wave number calculated using the formula 2 * pi / wavelength.
        """

        return 2 * np.pi / self.wavelength

    @property
    def pulsation(self):
        """
        The pulsation of the source, calculated from the wave number and speed of light.

        Returns
        -------
        float
            The pulsation of the source.
        """
        return self.wavenumber * c

    @property
    def frequency(self):
        """
        The frequency of the source, calculated from the wavelength.

        Returns
        -------
        float
            The frequency of the source.
        """
        return c / self.wavelength

    @abstractmethod
    def expression(self):
        """
        Compute the expression for the source.

        This method must be implemented by subclasses to provide the specific
        expression of the source in the framework.

        Returns
        -------
        dolfin.Expression
            The Dolfin expression representing the source.
        """

        pass

    def plot(self, figsize=None, ax=None):
        """
        Plot the source expression.

        Parameters
        ----------
        figsize : tuple of int, optional
            The size of the figure in inches. If None, the size of the figure is
            not modified.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot.
        ax : matplotlib.axes.Axes
            The axes containing the plot.
        tri : matplotlib.tri.Triangulation
            The triangulation of the mesh.
        cb : matplotlib.colorbar.Colorbar
            The colorbar of the plot.
        """
        if self.dim == 2:
            tri, cb = plot(self.expression, mesh=self.domain, ax=ax)
            fig = plt.gcf()
            ax = fig.axes
            ax_ = ax[:2]
            ax_[0].set_title("Re")
            ax_[1].set_title("Im")
            for a in ax_:
                a.set_xlabel("x")
                a.set_ylabel("y")
            if figsize is not None:
                fig.set_size_inches(figsize)
            plt.tight_layout()
            return fig, ax, tri, cb
        else:
            raise NotImplementedError("plot not implemented in 3D")
