#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


__all__ = [
    "array2function",
    "function2array",
    "project_iterative",
    "get_coordinates",
    "rot_matrix_2d",
    "tanh",
]


import numpy as np

from .. import dolfin
from ..complex import project


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
    dofmap = A.dofmap()
    x = A.tabulate_dof_coordinates().reshape((-1, 2))
    first_dof, last_dof = dofmap.ownership_range()
    unowned = dofmap.local_to_global_unowned()
    # dofs = [dofmap.local_to_global_index(dof) not in unowned for dof in range(last_dof-first_dof)]

    u.vector().set_local(a[first_dof:last_dof])
    # u.vector().set_local(a)

    # print("len(array2function)", len(a))
    # print("first last ", (first_dof, last_dof))
    # print("len(sub array)", len(a[first_dof:last_dof]))

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


def project_iterative(applied_function, function_space):
    return project(
        applied_function,
        function_space,
        solver_type="cg",
        preconditioner_type="jacobi",
    )


def get_coordinates(A):
    n = A.dim()
    d = A.mesh().geometry().dim()
    dof_coordinates = A.tabulate_dof_coordinates()
    dof_coordinates.resize((n, d))
    return dof_coordinates


def rot_matrix_2d(t):
    return np.array([[np.sin(t), -np.cos(t), 0], [np.cos(t), np.sin(t), 0], [0, 0, 1]])


def tanh(x):
    return (dolfin.exp(2 * x) - 1) / (dolfin.exp(2 * x) + 1)
