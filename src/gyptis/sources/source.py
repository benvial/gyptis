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


def vector(components):
    return (
        components[0] * _COORD.i + components[1] * _COORD.j + components[2] * _COORD.k
    )


x = sp.symbols("x[0] x[1] x[2]", real=True)
X = vector(np.array(x))


def expression2complex_2d(expr, **kwargs):
    re, im = (p.subs(x[2], 0) for p in expr.as_real_imag())
    dexpr = [dolfin.Expression(sp.printing.ccode(p), **kwargs) for p in (re, im)]
    return Complex(*dexpr)


class Source(ABC):
    def __init__(self, wavelength, dim, phase=0, amplitude=1, degree=1, domain=None):
        self.wavelength = wavelength
        self.phase = phase
        self.amplitude = amplitude
        self.dim = dim
        self.degree = degree
        self.domain = domain

    @property
    def wavenumber(self):
        return 2 * np.pi / self.wavelength

    @property
    def pulsation(self):
        return self.wavenumber * c

    @property
    def frequency(self):
        return c / self.wavelength

    @abstractmethod
    def expression(self):
        pass

    def plot(self, figsize=None, ax=None):
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
