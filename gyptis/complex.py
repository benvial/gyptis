#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

"""
Support for complex finite element forms.
This module provides a class::Complex class and overrides some `dolfin` functions 
to easily deal with complex problems by spliting real and imaginary parts.
"""

import numpy as np
import dolfin as df
import ufl


def complexcheck(func):
    """Wrapper to check if arguments are complex"""

    def wrapper(self, z):
        if hasattr(z, "real") and hasattr(z, "imag"):
            if not isinstance(z, Complex):
                z = Complex(z.real, z.imag)
        else:
            z = Complex(z, 0)
        return func(self, z)

    return wrapper


def complexify_lin(func):
    """Wrapper to check if arguments are complex"""

    def wrapper(z, *args, **kwargs):
        if isinstance(z, Complex):
            return Complex(func(z.real, *args, **kwargs), func(z.imag, *args, **kwargs))
        else:
            return func(z, *args, **kwargs)

    return wrapper


def complexify_vec(func):
    def wrapper(*args, **kwargs):
        v = func(*args, **kwargs)
        return Complex(v[0], v[1]), v

    return wrapper


# def complexify_bilin(func):
#     def wrapper(a,b, *args, **kwargs):
#         if isinstance(z, Complex):
#             return Complex(func(z.real, *args, **kwargs),func(z.imag, *args, **kwargs))
#         else:
#             return func(a,b, *args, **kwargs)
#     return wrapper


class Complex(object):
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __len__(self):
        if hasattr(self.real, "__len__"):
            return len(self.real)
        else:
            return 0

    def __iter__(self):
        for i in range(len(self)):
            yield Complex(self.real[i], self.imag[i])

    # def __next__(self):
    #     i = self.i
    #     self.i += 1
    #     yield Complex(self.real[i], self.imag[i])
    #
    #

    @complexcheck
    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other):
        return self.__add__(other)

    @complexcheck
    def __sub__(self, other):
        return Complex(self.real - other.real, self.imag - other.imag)

    def __rsub__(self, other):
        return self.__sub__(other)

    @complexcheck
    def __mul__(self, other):
        return Complex(
            self.real * other.real - self.imag * other.imag,
            self.imag * other.real + self.real * other.imag,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    @complexcheck
    def __truediv__(self, other):
        sr, si, tr, ti = self.real, self.imag, other.real, other.imag  # short forms
        r = tr ** 2 + ti ** 2
        return Complex((sr * tr + si * ti) / r, (si * tr - sr * ti) / r)

    @complexcheck
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

    def __abs__(self):
        return df.sqrt(self.real ** 2 + self.imag ** 2)

    def __neg__(self):  # defines -c (c is Complex)
        return Complex(-self.real, -self.imag)

    @complexcheck
    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    @complexcheck
    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"({self.real.__str__()} + {self.imag.__str__()}j)"

    def __repr__(self):
        return "Complex" + f"({self.real.__repr__()}, {self.imag.__repr__()})"

    def __pow__(self, power):
        if iscomplex(power):
            raise NotImplementedError("complex exponent not implemented")
        else:
            A, phi = self.polar()
            return self.polar2cart((A ** power, phi * power))

    def phase(self):
        y, x = self.imag, self.real
        
        
        

        return df.conditional(
            ufl.eq(self.__abs__(), 0),
            0,
            df.conditional(
                ufl.eq(self.__abs__() + x, 0),
                df.pi,
                2 * df.atan(y / (self.__abs__() + x)),
            ),
        )

    @staticmethod
    def polar2cart(t):
        A, phi = t
        # if A<0:
        #     raise ValueError("amplitude must be positive")
        return A * Complex(df.cos(phi), df.sin(phi))

    def polar(self):
        return self.__abs__(), self.phase()


def iscomplex(z):
    if hasattr(z, "real") and hasattr(z, "imag") and not np.all(z.imag == 0):
        return True
    else:
        return False


def dotc(a, b):
    if iscomplex(a) and iscomplex(b):
        re = df.dot(a.real, b.real) - df.dot(a.imag, b.imag)
        im = df.dot(a.real, b.imag) + df.dot(a.imag, b.real)
        return Complex(re, im)
    elif iscomplex(a) and not iscomplex(b):
        re = df.dot(a.real, b)
        im = df.dot(a.imag, b)
        return Complex(re, im)
    elif not iscomplex(a) and iscomplex(b):
        re = df.dot(a, b.real)
        im = df.dot(a, b.imag)
        return Complex(re, im)
    else:
        return df.dot(a, b)


def innerc(a, b):
    if iscomplex(a) and iscomplex(b):
        re = df.inner(a.real, b.real) - df.inner(a.imag, b.imag)
        im = df.inner(a.real, b.imag) + df.inner(a.imag, b.real)
        return Complex(re, im)
    elif iscomplex(a) and not iscomplex(b):
        re = df.inner(a.real, b)
        im = df.inner(a.imag, b)
        return Complex(re, im)
    elif not iscomplex(a) and iscomplex(b):
        re = df.inner(a, b.real)
        im = df.inner(a, b.imag)
        return Complex(re, im)
    else:
        return df.inner(a, b)


assemblec = complexify_lin(df.assemble)

Functionc = complexify_vec(df.Function)
TrialFunctionc = complexify_vec(df.TrialFunction)
TestFunctionc = complexify_vec(df.TestFunction)
gradc = complexify_lin(df.grad)


if __name__ == "__main__":

    x, y, x1, y1 = np.random.rand(4) - 0.5

    tol = 1e-15

    # x, y = 1, 2
    # x1, y1 = 12, -4
    z = Complex(x, y)
    z1 = Complex(x1, y1)
    q = x + 1j * y
    q1 = x1 + 1j * y1
    assert z == q
    assert -z == Complex(-x, -y)
    assert z != Complex(x / 2, y)
    assert z != Complex(x, y / 2)
    assert z != Complex(x / 2, y / 2)
    assert z != Complex(x, y) / 2
    assert z != q / q1
    assert z != q / 2
    assert 2 * z == Complex(2 * x, 2 * y)
    assert z / z == 1
    assert 1 / z == 1 / z
    assert z * z1 == q * q1
    assert z * q1 == q * z1
    assert z / q1 == q / z1
    # assert z/z1 == q/q1
    assert np.abs(z / z1 - q / q1) < tol
    assert abs(z) == (x ** 2 + y ** 2) ** 0.5
    assert z.conj == q.conj()

    nx, ny = 11, 11
    mesh_model = df.UnitSquareMesh(nx, ny)
    W = df.FunctionSpace(mesh_model, "CG", 1)
    W0 = df.FunctionSpace(mesh_model, "DG", 0)
    # W0 = df.FunctionSpace(mesh_model, "CG", 1)

    ure = df.Function(W)
    uim = df.Function(W)

    u = Complex(ure, uim)

    a = df.Expression("12", degree=2)
    u = Complex(-1 * 0, -0)

    test = u
    test = u.phase()
    dx = df.dx(domain=mesh_model)

    astest = assemblec(test * dx)
    print(astest)

    xas
    import matplotlib.pyplot as plt

    plt.ion()
    plt.clf()
    cm = df.plot(df.project(are, W), cmap="RdBu")
    plt.colorbar(cm)

    csd

    nx, ny = 100, 100
    mesh_model = df.UnitSquareMesh(nx, ny)

    # real
    A = df.FunctionSpace(mesh_model, "CG", 1)
    a = df.Expression("x[0]", degree=2)
    b = df.project(a, A)

    s = df.inner(b, b)
    k = df.assemble(s * df.dx)

    print(k)

    ## test complex
    z = Complex(b, b)

    z + b

    V = df.Constant((1, 0))
    V = Complex(V, V * 2)
    dc = dotc(V, df.grad(b))

    print(dc)
    k = assemblec(dc * df.dx)

    print(k)

    k = assemblec(z * df.dx)

    print(k)

    Vel = df.FiniteElement("P", "triangle", 1)
    Ac = df.FunctionSpace(mesh_model, Vel * Vel)

    # a = Expression("(x[0], 0)", degree=2)
    a = df.Expression(("x[0]-x[1]", "3*x[0]+2*x[1]"), degree=2)
    b = df.Expression(("11*x[0]*x[1]", "0.1*x[1]+4*x[0]"), degree=2)

    # a = Expression("2", degree=1)
    # a = Complex(a[0],a[1])
    ap = df.project(a, Ac)
    bp = df.project(b, Ac)

    ac = Complex(ap[0], ap[1])
    bc = Complex(bp[0], bp[1])

    t = ac * bc * df.dx
    print(assemblec(t))

    gradc = complexify_lin(df.grad)

    t = dotc(gradc(ac), gradc(bc)) * df.dx
    print(assemblec(t))
    t = innerc(gradc(ac), gradc(bc)) * df.dx
    print(assemblec(t))

    x, y = np.linspace(0, 1, 10), np.linspace(0, 1, 10)

    z = Complex(x, y)

    print(len(z))

    for j in z:
        print(j.__repr__())
        print("-" * 55)

    #
    #
    # ### complex expression
    # import sympy as sp
    # import numpy as np
    # from sympy.vector import CoordSys3D
    #
    # N = CoordSys3D("N")
    # # v1 = 2 * N.i + 3 * N.j - N.k
    # # v2 = N.i - 4 * N.j + N.k
    # # v1.dot(v2)
    # # v1.cross(v2)
    # # # Alternately, can also do
    # # v1 & v2
    # # v1 ^ v2
    #
    # vector = lambda x: x[0]*N.i + x[1]*N.j + x[2]*N.k
    # x = sp.symbols("x[0] x[1] x[2]", real=True)
    # x = np.array(x)
    # X = vector(x)
    # theta= np.pi/6
    # Kdir = vector((np.cos(theta),np.sin(theta),0.1))
    # lambda0=0.1
    # k0 = 2*np.pi/lambda0
    # K = Kdir*k0
    # X0 = 0.5,0.5,1
    # gamma = 1111,0.2,1
    #
    # Xp = (x-X0)/np.array(gamma)
    #
    # # Xp -= np.array(list(Kdir.components.values()))
    #
    # expr = sp.exp(1j*K.dot(X) -Xp.dot(Xp))
    # re, im = expr.as_real_imag()
    #
    # # 2D: x[2] fails!
    # re = re.subs(x[2],0)
    #
    # # s = "x[0] + x[1]"
    # # sp.parse_expr(s)
    #
    # code = sp.printing.ccode(re)
    #
    #
    #
    # are = df.Expression(code, degree=2, k=0)
    #
    #
    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.clf()
    # cm = df.plot(df.project(are,W),cmap="RdBu")
    # plt.colorbar(cm)
    #
    # s = "x[0] +x[1] +x[2]"
    #
    # xas
