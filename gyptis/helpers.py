#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import sys

import numpy as np

from . import dolfin

_DirichletBC = dolfin.DirichletBC
_Measure = dolfin.Measure


def list_time():
    return dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])


def _get_form(u):
    form = 0
    for f in u:
        try:
            form += f.form
        except:
            return
    return form


def get_coords(A):
    n = A.dim()
    d = A.mesh().geometry().dim()
    dof_coordinates = A.tabulate_dof_coordinates()
    dof_coordinates.resize((n, d))
    return dof_coordinates


class Measure(_Measure):
    def __init__(
        self,
        integral_type,
        domain=None,
        subdomain_id="everywhere",
        metadata=None,
        subdomain_data=None,
        subdomain_dict=None,
    ):

        self.subdomain_dict = subdomain_dict
        if (
            self.subdomain_dict
            and isinstance(subdomain_id, str)
            and subdomain_id != "everywhere"
        ):
            subdomain_id = self.subdomain_dict[subdomain_id]
        super().__init__(
            integral_type,
            domain=domain,
            subdomain_id=subdomain_id,
            metadata=metadata,
            subdomain_data=subdomain_data,
        )

    def __call_single__(self, subdomain_id=None, **kwargs):
        if (
            self.subdomain_dict
            and isinstance(subdomain_id, str)
            and subdomain_id != "everywhere"
        ):
            subdomain_id = self.subdomain_dict[subdomain_id]
        return super().__call__(subdomain_id=subdomain_id, **kwargs)

    def __call__(self, subdomain_id=None, **kwargs):
        if isinstance(subdomain_id, list):
            for i, sid in enumerate(subdomain_id):
                if i == 0:
                    out = self.__call_single__(subdomain_id=sid, **kwargs)
                else:
                    out += self.__call_single__(subdomain_id=sid, **kwargs)
            return out
        else:
            return self.__call_single__(subdomain_id=subdomain_id, **kwargs)


class DirichletBC(_DirichletBC):
    def __init__(self, *args, **kwargs):
        self.subdomain_dict = args[-1]
        if not callable(args[2]):
            args = list(args)
            args[-2] = self.subdomain_dict[args[-2]]
            args = tuple(args[:-1])
        super().__init__(*args)


#
# tol = DOLFIN_EPS
# parameters["krylov_solver"]["error_on_nonconvergence"] = False
# parameters["form_compiler"]["cpp_optimize"] = True
# parameters["form_compiler"]["representation"] = "uflacs"
# parameters["form_compiler"]["quadrature_degree"] = 2
# parameters["allow_extrapolation"] = True


def rot_matrix_2d(t):
    return np.array([[np.sin(t), -np.cos(t), 0], [np.cos(t), np.sin(t), 0], [0, 0, 1]])


def make_unit_vectors(dim):
    if dim == 3:
        ex = Constant((1, 0, 0))
        ey = Constant((0, 1, 0))
        ez = Constant((0, 0, 1))
        return [ex, ey, ez]
    else:
        ex = Constant((1, 0))
        ey = Constant((0, 1))
        return [ex, ey]


def mpi_print(s):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(s)
        sys.stdout.flush()


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


def matfmt(m, ndigit=4, extra_space=0, cplx=False):
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


def sgn(u):
    if np.sign(u) == -1:
        return "-"
    else:
        return "+"


def printim(y):
    return f"{sgn(float(y))}{abs(float(y))}j"


def tanh(x):
    return (exp(2 * x) - 1) / (exp(2 * x) + 1)


#
# #### geometry
#
#
# def boundary_L(x, on_boundary):
#     return on_boundary and (dolfin.near(x[1], 0, tol))
#
#
# def boundary_R(x, on_boundary):
#     return on_boundary and (dolfin.near(x[1], 1, tol))
#
#
# class InnerBoundary(SubDomain):
#     """
#     The inner boundaries of the mesh
#     """
#
#     def inside(self, x, on_boundary):
#         return (
#             x[0] > DOLFIN_EPS
#             and x[0] < 1 - DOLFIN_EPS
#             and x[1] > DOLFIN_EPS
#             and x[1] < 1 - DOLFIN_EPS
#             and on_boundary
#         )
#
#
#
class PeriodicBoundary2DX(dolfin.SubDomain):
    def __init__(self, period, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period

    def inside(self, x, on_boundary):
        return bool(dolfin.near(x[0], -self.period / 2) and on_boundary)

    # # Left boundary is "target domain" G
    # def inside(self, x, on_boundary):
    #     return bool(
    #         x[0] - self.period / 2 < dolfin.DOLFIN_EPS
    #         and x[0] - self.period / 2 > -dolfin.DOLFIN_EPS
    #         and on_boundary
    #     )

    def map(self, x, y):
        y[0] = x[0] - self.period
        y[1] = x[1]

    #
    # def map(self, x, y):
    #     if dolfin.near(x[0], self.period / 2):
    #         y[0] = x[0] - self.period
    #         y[1] = x[1]
    #     else:
    #         y[0] = -1000
    #         y[1] = -1000


class BiPeriodicBoundary3D(dolfin.SubDomain):
    def __init__(self, period, **kwargs):
        self.period = period
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        return bool(
            (
                dolfin.near(x[0], -self.period[0] / 2)
                or dolfin.near(x[1], -self.period[1] / 2)
            )
            and (
                not (
                    (
                        dolfin.near(x[0], -self.period[0] / 2)
                        and dolfin.near(x[1], self.period[1] / 2)
                    )
                    or (
                        dolfin.near(x[0], self.period[0] / 2)
                        and dolfin.near(x[1], -self.period[1] / 2)
                    )
                )
            )
            and on_boundary
        )

    def map(self, x, y):

        if dolfin.near(x[0], self.period[0] / 2):
            y[0] = x[0] - self.period[0]
            y[1] = x[1]
            y[2] = x[2]
        elif dolfin.near(x[1], self.period[1] / 2):
            y[0] = x[0]
            y[1] = x[1] - self.period[1]
            y[2] = x[2]
        else:
            y[0] = -1000
            y[1] = -1000
            y[2] = -1000


# class DomainBoundary(InnerBoundary):
#     def __init__(self, geom, *args, tol=1e-6, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.geom = geom
#         self.p1 = geom.first_corner().array()
#         self.p2 = geom.second_corne().array()
#         self.tol = tol
#
#     #
#     # def inside(self, x, on_boundary):
#     #     return (
#     #         on_boundary
#     #         and x[0] > self.p1[0] - self.tol
#     #         and x[0] < self.p2[0] + self.tol
#     #         and x[1] > self.p1[1] - self.tol
#     #         and x[1] < self.p2[1] + self.tol
#     #     )
#
#     def inside(self, x, on_boundary):
#         inside_rect = (
#             x[0] > self.p1[0] - self.tol
#             and x[0] < self.p2[0] + self.tol
#             and x[1] > self.p1[1] - self.tol
#             and x[1] < self.p2[1] + self.tol
#         )
#         outside_rect = (
#             x[0] < self.p1[0] + self.tol
#             or x[0] > self.p2[0] - self.tol
#             or x[1] < self.p1[1] + self.tol
#             or x[1] > self.p2[1] - self.tol
#         )
#         return inside_rect and outside_rect
