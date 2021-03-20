#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import sys

import numpy as np

from . import dolfin


def array2function(a, A):
    """Convert a numpy array to a fenics Function.

    Parameters
    ----------
    a : numpy array
        The array to convert.
    A : FunctionSpace
        The function space to interpolate on.

    Returns
    -------
    Function
        The converted array.

    """
    u = dolfin.Function(A)
    u.vector().set_local(a)
    dolfin.as_backend_type(u.vector()).update_ghost_values()
    u.vector().apply("insert")
    return u


def function2array(f):
    """Convert a fenics Function to a numpy array.

    Parameters
    ----------
    f : Function
        The function to convert.

    Returns
    -------
    numpy array
        The converted function.

    """
    return f.vector().get_local()


def list_time():
    return dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])


def get_coordinates(A):
    n = A.dim()
    d = A.mesh().geometry().dim()
    dof_coordinates = A.tabulate_dof_coordinates()
    dof_coordinates.resize((n, d))
    return dof_coordinates


def mpi_print(s):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(s)
        sys.stdout.flush()


def matfmt(m, ndigit=4, extra_space=0, cplx=False):
    def printim(y):
        return f"{sgn(float(y))}{abs(float(y))}j"

    def sgn(u):
        if np.sign(u) == -1:
            return "-"
        else:
            return "+"

    dim = len(m[0])

    pad = " " * extra_space

    if cplx:
        m = [[_.real, _.imag] for _ in np.ravel(m)]

    a = [f"%.{ndigit}f" % elem for elem in np.ravel(m)]

    if dim == 3:
        if cplx:
            b = f"""[{a[0]}{printim(a[1])} {a[2]}{printim(a[3])} 
            {a[4]}{printim(a[5])}] \n{pad}[{a[6]}{printim(a[7])} 
            {a[8]}{printim(a[9])} {a[10]}{printim(a[11])}] 
            \n{pad}[{a[12]}{printim(a[13])} {a[14]}{printim(a[15])} 
            {a[16]}{printim(a[17])}]
            """

        else:
            b = f"""[{a[0]} {a[1]} {a[2]}] \n{pad}[{a[3]} {a[4]} {a[5]}]
             \n{pad}[{a[6]} {a[7]} {a[8]}]
             """
    else:
        if cplx:
            b = f"""[{a[0]}{printim(a[1])} {a[2]}{printim(a[3])}] 
            \n{pad}[{a[4]}{printim(a[5])} {a[6]}{printim(a[7])}]
            """
        else:
            b = f"[{a[0]} {a[1]}] \n{pad}[{a[2]} {a[3]}]"
    return b


def matprint(*args, **kwargs):
    mpi_print(matfmt(*args, **kwargs))


def rot_matrix_2d(t):
    return np.array([[np.sin(t), -np.cos(t), 0], [np.cos(t), np.sin(t), 0], [0, 0, 1]])


def tanh(x):
    return (dolfin.exp(2 * x) - 1) / (dolfin.exp(2 * x) + 1)


def _translation_matrix(t):
    M = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    M[3], M[7], M[11] = t
    return M
