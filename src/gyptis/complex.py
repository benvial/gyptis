#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

"""
Support for complex finite element forms.
This module provides a :class:`Complex` class and overrides some ``dolfin`` functions
to easily deal with complex problems by spliting real and imaginary parts.
"""

from typing import Iterable

import numpy as np
import ufl

from . import dolfin


def _complexcheck(func):
    """Wrapper to check if arguments are complex"""

    def wrapper(self, z):
        if hasattr(z, "real") and hasattr(z, "imag"):
            if not isinstance(z, Complex):
                z = Complex(z.real, z.imag)
        else:
            z = Complex(z, 0)
        return func(self, z)

    return wrapper


def _complexify_linear(func):
    def wrapper(z, *args, **kwargs):
        if isinstance(z, Complex):
            return Complex(func(z.real, *args, **kwargs), func(z.imag, *args, **kwargs))
        else:
            return func(z, *args, **kwargs)

    return wrapper


def _complexify_bilinear(func):
    def wrapper(a, b, *args, **kwargs):
        if iscomplex(a) and iscomplex(b):
            re = func(a.real, b.real, *args, **kwargs) - func(
                a.imag, b.imag, *args, **kwargs
            )
            im = func(a.real, b.imag, *args, **kwargs) + func(
                a.imag, b.real, *args, **kwargs
            )
            return Complex(re, im)
        elif iscomplex(a) and not iscomplex(b):
            re = func(a.real, b, *args, **kwargs)
            im = func(a.imag, b)
            return Complex(re, im, *args, **kwargs)
        elif not iscomplex(a) and iscomplex(b):
            re = func(a, b.real, *args, **kwargs)
            im = func(a, b.imag)
            return Complex(re, im, *args, **kwargs)
        else:
            return func(a, b, *args, **kwargs)

    return wrapper


def _complexify_vector(func):
    def wrapper(*args, **kwargs):
        v = func(*args, **kwargs)
        re, im = dolfin.split(v)
        return Complex(re, im)

    return wrapper


def _complexify_vector_alt(func):
    def wrapper(*args, **kwargs):
        v = func(*args, **kwargs)
        re, im = v.split()  # (deepcopy=True)
        return Complex(re, im)

    return wrapper


class Complex:
    """A complex object.

    Parameters
    ----------
    real : type
        Real part.
    imag : type
        Imaginary part (the default is 0.0).

    Attributes
    ----------
    real
    imag

    """

    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __len__(self):
        if hasattr(self.real, "__len__") and hasattr(self.imag, "__len__"):
            if len(self.real) == len(self.imag):
                return len(self.real)
            else:
                raise ValueError("real and imaginary parts should have the same length")
        else:
            return 0

    def __iter__(self):
        for i in range(len(self)):
            yield Complex(self.real[i], self.imag[i])

    def __getitem__(self, i):
        return Complex(self.real[i], self.imag[i])

    @_complexcheck
    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    __radd__ = __add__

    @_complexcheck
    def __sub__(self, other):
        return Complex(self.real - other.real, self.imag - other.imag)

    __rsub__ = __sub__

    @_complexcheck
    def __mul__(self, other):
        return Complex(
            self.real * other.real - self.imag * other.imag,
            self.imag * other.real + self.real * other.imag,
        )

    __rmul__ = __mul__

    __array_ufunc__ = None

    @_complexcheck
    def __truediv__(self, other):
        sr, si, tr, ti = self.real, self.imag, other.real, other.imag  # short forms
        r = tr ** 2 + ti ** 2
        return Complex((sr * tr + si * ti) / r, (si * tr - sr * ti) / r)

    @_complexcheck
    def __rtruediv__(self, other):
        sr, si, tr, ti = other.real, other.imag, self.real, self.imag  # short forms
        r = tr ** 2 + ti ** 2
        return Complex((sr * tr + si * ti) / r, (si * tr - sr * ti) / r)

    @property
    def shape(self):
        return self.real.shape

    @property
    def conj(self):
        return Complex(self.real, -self.imag)

    @property
    def module(self):
        """Module of the complex number"""
        return self.__abs__()

    @property
    def phase(self):
        return self.__angle__()

    def __abs__(self):
        return dolfin.sqrt(self.real ** 2 + self.imag ** 2)

    def __neg__(self):  # defines -c (c is Complex)
        return Complex(-self.real, -self.imag)

    @_complexcheck
    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    @_complexcheck
    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"({self.real.__str__()} + {self.imag.__str__()}j)"

    def __repr__(self):
        return "Complex" + f"({self.real.__repr__()}, {self.imag.__repr__()})"

    def __pow__(self, power):
        if iscomplex(power) and power.imag != 0:
            raise NotImplementedError("complex exponent not implemented")
        else:
            A, phi = self.polar()
            return self.polar2cart(A ** power, phi * power)

    def __angle__(self):
        x, y = self.real, self.imag
        try:
            return np.angle(x + 1j * y)
        except:
            return dolfin.conditional(
                ufl.eq(self.__abs__(), 0),
                0,
                dolfin.conditional(
                    ufl.eq(self.__abs__() + x, 0),
                    dolfin.pi,
                    2 * dolfin.atan(y / (self.__abs__() + x)),
                ),
            )

    def __call__(self, *args, **kwargs):
        "Calls the complex function if base objects are callable"
        return Complex(
            self.real.__call__(*args, **kwargs),
            self.imag.__call__(*args, **kwargs),
        )

    def tocomplex(self):
        return self.real + 1j * self.imag

    @staticmethod
    def polar2cart(module, phase):
        """Polar to cartesian representation.

        Parameters
        ----------
        module : type
            The module (positive).
        phase : type
            The polar angle.

        Returns
        -------
        Complex
            The complex number in cartesian representation.

        """
        return module * Complex(dolfin.cos(phase), dolfin.sin(phase))

    def polar(self):
        """Polar representation.

        Returns
        -------
        tuple
            Modulus and phase.

        """
        return self.__abs__(), self.__angle__()

    def to_complex(self):
        return self.real + 1j * self.imag


def iscomplex(z):
    """Checks if object is complex.

    Parameters
    ----------
    z : type
        Object.

    Returns
    -------
    bool
        True if z is complex, else False.

    """
    if hasattr(z, "real") and hasattr(z, "imag") and z.imag is not None:
        return True
    else:
        return False


class ComplexFunctionSpace(dolfin.FunctionSpace):
    """Complex function space"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        element = super().ufl_element()
        super().__init__(super().mesh(), element * element, **kwargs)


def _cplx_iter(f):
    def wrapper(v, *args, **kwargs):
        iterable = isinstance(v, Iterable)
        cplx = any([iscomplex(v_) for v_ in v]) if iterable else iscomplex(v)
        if cplx:
            if iterable:
                v = np.array(v)
            return Complex(f(v.real, *args, **kwargs), f(v.imag, *args, **kwargs))
        else:
            return f(v, *args, **kwargs)

    return wrapper


def phasor(prop_cst, direction=0, **kwargs):
    phasor_re = dolfin.Expression(
        f"cos(prop_cst*x[{direction}])", prop_cst=prop_cst, **kwargs
    )
    phasor_im = dolfin.Expression(
        f"sin(prop_cst*x[{direction}])", prop_cst=prop_cst, **kwargs
    )
    return Complex(phasor_re, phasor_im)


def phase_shift(phase, **kargs):
    phasor_re = dolfin.Expression(f"cos(phase)", phase=phase, **kargs)
    phasor_im = dolfin.Expression(f"sin(phase)", phase=phase, **kargs)
    return Complex(phasor_re, phasor_im)


def vector(vect):
    vsr = [v.real for v in vect]
    vsi = [v.imag for v in vect]
    return Complex(dolfin.as_vector(vsr), dolfin.as_vector(vsi))


def tensor(tens):
    tsr = [[t.real for t in _t] for _t in tens]
    tsi = [[t.imag for t in _t] for _t in tens]
    return Complex(dolfin.as_tensor(tsr), dolfin.as_tensor(tsi))


interpolate = _complexify_linear(dolfin.interpolate)
assemble = _complexify_linear(dolfin.assemble)
grad = _complexify_linear(dolfin.grad)
div = _complexify_linear(dolfin.div)
curl = _complexify_linear(dolfin.curl)
project = _complexify_linear(dolfin.project)

Dx = _complexify_linear(dolfin.Dx)
inner = _complexify_bilinear(dolfin.inner)
dot = _complexify_bilinear(dolfin.dot)
cross = _complexify_bilinear(dolfin.cross)
as_tensor = _cplx_iter(dolfin.as_tensor)
as_vector = _cplx_iter(dolfin.as_vector)
Constant = _cplx_iter(dolfin.Constant)
Function = _complexify_vector_alt(dolfin.Function)
TrialFunction = _complexify_vector(dolfin.TrialFunction)
TestFunction = _complexify_vector(dolfin.TestFunction)
TrialFunctions = _complexify_vector(dolfin.TrialFunctions)
TestFunctions = _complexify_vector(dolfin.TestFunctions)
j = Complex(0, 1)


def _traverse(a):
    if not isinstance(a, list):
        yield a
    else:
        for e in a:
            yield from _traverse(e)


def to_array(a):
    l = []
    for e in _traverse(a):
        l.append(e.to_complex())
    return np.array(l).reshape(np.array(a).shape[:-1])
