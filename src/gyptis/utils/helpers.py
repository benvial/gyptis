#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


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


def array2function(values, function_space):
    """Convert a numpy array to a fenics Function.

    Parameters
    ----------
    values : numpy array
        The array to convert.
    function_space : FunctionSpace
        The function space to interpolate on.

    Returns
    -------
    Function
        The converted array.

    """

    # print(len(values))

    u = dolfin.Function(function_space)
    dofmap = function_space.dofmap()
    # order = dofmap.tabulate_local_to_global_dofs()

    dofmap = function_space.dofmap()
    my_first, my_last = dofmap.ownership_range()
    sub_array = values[my_first:my_last]  # [order]
    # print("order", len(order))
    u.vector().set_local(sub_array)
    dolfin.as_backend_type(u.vector()).update_ghost_values()
    # dolfin.as_backend_type(u.vector()).vec().ghostUpdate()
    # u.vector().apply("")
    u.vector().apply("insert")
    return u


#
# def array2function(values, function_space):
#     u = dolfin.Function(function_space)
#     u.set_allow_extrapolation(True)
#     u.vector().apply("insert")
#     u.vector().set_local(values)
#     dolfin.as_backend_type(u.vector()).update_ghost_values()
#     u.vector().apply("insert")
#     return u
#
#
#
# def array2function(values, function_space):
#     V = function_space
#     u = dolfin.Function(function_space)
#     vec = u.vector()
#     mesh = function_space.mesh
#
#     dofmap = V.dofmap()
#     my_first, my_last = dofmap.ownership_range()                # global
#
#     # # 'Handle' API change of tabulate coordinates
#     # if df.dolfin_version().split('.')[1] == '7':
#     #     x = V.tabulate_dof_coordinates().reshape((-1, 2))
#     # else:
#     # x = V.dofmap().tabulate_all_coordinates(mesh)
#     #
#     x = V.tabulate_dof_coordinates().reshape((-1, 2))
#
#     unowned = dofmap.local_to_global_unowned()
#     dofs = [dofmap.local_to_global_index(dof) not in unowned for dof in range(my_last-my_first)]
#     x = x[dofs]
#     vec.set_local(values)
#     vec.apply('insert')
#     return u


def function2array(f, space=None):
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
    values = f.vector().get_local()
    function_space = f.function_space()
    dofmap = function_space.dofmap()
    my_first, my_last = dofmap.ownership_range()
    sub_array = values  # [my_first:my_last]

    return sub_array


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
