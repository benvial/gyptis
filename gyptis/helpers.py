#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import time
import sys
import os
from functools import cached_property
import numpy as np
from copy import copy
from scipy.interpolate import griddata
from dolfin import Measure as __Measure__
from dolfin import DirichletBC as __DirichletBC__
import dolfin as df
import meshio




class Measure(__Measure__):
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


class DirichletBC(__DirichletBC__):
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
    if MPI.rank(MPI.comm_world) == 0:
        print(s)
        sys.stdout.flush()


def array2mesh(a, A):
    u = Function(A)
    u.vector().set_local(a)
    a = u
    as_backend_type(a.vector()).update_ghost_values()
    a.vector().apply("insert")
    return a


def mesh2array(a):
    return a.vector().get_local()


def matfmt(m, ndigit=4, extra_space=0, cplx=False):
    dim = len(m[0])

    pad = " " * extra_space

    if cplx:
        m = [[_.real, _.imag] for _ in np.ravel(m)]

    a = [f"%.{ndigit}f" % elem for elem in np.ravel(m)]

    if dim == 3:
        if cplx:
            b = f"[{a[0]}{printim(a[1])} {a[2]}{printim(a[3])} {a[4]}{printim(a[5])}] \n{pad}[{a[6]}{printim(a[7])} {a[8]}{printim(a[9])} {a[10]}{printim(a[11])}] \n{pad}[{a[12]}{printim(a[13])} {a[14]}{printim(a[15])} {a[16]}{printim(a[17])}]"

        else:
            b = f"[{a[0]} {a[1]} {a[2]}] \n{pad}[{a[3]} {a[4]} {a[5]}] \n{pad}[{a[6]} {a[7]} {a[8]}]"
    else:
        if cplx:
            b = f"[{a[0]}{printim(a[1])} {a[2]}{printim(a[3])}] \n{pad}[{a[4]}{printim(a[5])} {a[6]}{printim(a[7])}]"
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
#     return on_boundary and (near(x[1], 0, tol))
# 
# 
# def boundary_R(x, on_boundary):
#     return on_boundary and (near(x[1], 1, tol))
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
# class PeriodicBoundary2D(SubDomain):
# 
#     # Left boundary is "target domain" G
#     def inside(self, x, on_boundary):
#         # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
#         return bool(
#             (near(x[0], 0) or near(x[1], 0))
#             and (
#                 not (
#                     (near(x[0], 0) and near(x[1], 1))
#                     or (near(x[0], 1) and near(x[1], 0))
#                 )
#             )
#             and on_boundary
#         )
# 
#     def map(self, x, y):
#         if near(x[0], 1) and near(x[1], 1):
#             y[0] = x[0] - 1.0
#             y[1] = x[1] - 1.0
#         elif near(x[0], 1):
#             y[0] = x[0] - 1.0
#             y[1] = x[1]
#         else:  # near(x[1], 1)
#             y[0] = x[0]
#             y[1] = x[1] - 1.0
# 
# 
# class PeriodicBoundary3D(SubDomain):
# 
#     # Left boundary is "target domain" G
#     def inside(self, x, on_boundary):
#         # return True if on left or bottom boundary AND NOT on one of the two slave edges
#         return bool(
#             (near(x[0], 0) or near(x[1], 0) or near(x[2], 0))
#             and (
#                 not (
#                     (near(x[0], 1) and near(x[2], 0))
#                     or (near(x[0], 0) and near(x[2], 1))
#                     or (near(x[1], 1) and near(x[2], 0))
#                     or (near(x[1], 0) and near(x[2], 1))
#                 )
#             )
#             and on_boundary
#         )
# 
#     # Map right boundary (H) to left boundary (G)
#     def map(self, x, y):
# 
#         if near(x[0], 1) and near(x[2], 1):
#             y[0] = x[0] - 1
#             y[1] = x[1]
#             y[2] = x[2] - 1
#         elif near(x[1], 1) and near(x[2], 1):
#             y[0] = x[0]
#             y[1] = x[1] - 1
#             y[2] = x[2] - 1
#         elif near(x[0], 1):
#             y[0] = x[0] - 1
#             y[1] = x[1]
#             y[2] = x[2]
#         elif near(x[1], 1):
#             y[0] = x[0]
#             y[1] = x[1] - 1
#             y[2] = x[2]
#         elif near(x[2], 1):
#             y[0] = x[0]
#             y[1] = x[1]
#             y[2] = x[2] - 1
#         else:
#             y[0] = -1000
#             y[1] = -1000
#             y[2] = -1000
# 
# 
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


class _PermittivityPy(df.UserExpression):
    def __init__(self, markers, subdomains, value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.markers = markers
        self.subdomains = subdomains
        self.value = value

    def eval_cell(self, values, x, cell):
        for sub, val in self.value.items():
            if self.markers[cell.index] == self.subdomains[sub]:
                if callable(val):
                    values[:] = val(x)
                else:
                    values[:] = val
                # values[:] = val

    def value_shape(self):
        return ()


class _PermittivityCpp(df.CompiledExpression):
    def __init__(self, markers, subdomains, value, **kwargs):
        with open("epsilon.cpp") as f:
            permittivity_code = f.read()
        compiled_cpp = df.compile_cpp_code(permittivity_code).PermittivityCpp()
        super().__init__(
            compiled_cpp, markers=markers, subdomains=subdomains, value=value, **kwargs
        )
        self.markers = markers
        self.subdomains = subdomains
        self.value = value


class Permittivity(object):
    def __new__(self, markers, subdomains, value, cpp=True, **kwargs):
        if cpp:
            return _PermittivityCpp(markers, subdomains, value, **kwargs)
        else:
            return _PermittivityPy(markers, subdomains, value, **kwargs)


def read_mesh(mesh_file, data_dir="./data"):
    t = -time.time()
    msh = meshio.read(mesh_file)
    t += time.time()
    mpi_print(f"      meshio time: {t:0.3f}s")
    

    face_cells = []
    for cell in msh.cells:
        if cell.type == "tetra":
            tetra_cells = cell.data
        elif cell.type == "triangle":
            if len(face_cells) == 0:
                face_cells = cell.data
            else:
                face_cells = np.vstack([face_cells, cell.data])

    face_data = []
    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "triangle":
            if len(face_data) == 0:
                face_data = msh.cell_data_dict["gmsh:physical"][key]
            else:
                face_data = np.vstack(
                    [face_data, msh.cell_data_dict["gmsh:physical"][key]]
                )
        elif key == "tetra":
            tetra_data = msh.cell_data_dict["gmsh:physical"][key]

    tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells},)
    tetra_mesh_data = meshio.Mesh(
        points=msh.points,
        cells={"tetra": tetra_cells},
        cell_data={"subdomain": [tetra_data]},
    )
    triangle_mesh_data = meshio.Mesh(
        points=msh.points,
        cells={"triangle": face_cells},
        cell_data={"subdomain": [face_data]},
    )
    
    
    t = -time.time()
    meshio.write(f"{data_dir}/mesh.xdmf", tetra_mesh)
    meshio.xdmf.write(f"{data_dir}/mf.xdmf", tetra_mesh_data)
    meshio.xdmf.write(f"{data_dir}/mf_surf.xdmf", triangle_mesh_data)
    t += time.time()
    mpi_print(f"      xdmf write time: {t:0.3f}s")
    t = -time.time()
    ### markers
    mesh_model = Mesh()
    with XDMFFile(f"{data_dir}/mesh.xdmf") as infile:
        infile.read(mesh_model)
    mvc = MeshValueCollection("size_t", mesh_model, 3)
    with XDMFFile(f"{data_dir}/mf.xdmf") as infile:
        infile.read(mvc, "subdomain")
    markers = cpp.mesh.MeshFunctionSizet(mesh_model, mvc)
    mvc_surf = MeshValueCollection("size_t", mesh_model, 2)
    with XDMFFile(f"{data_dir}/mf_surf.xdmf") as infile:
        infile.read(mvc_surf, "subdomain")
    markers_surf = cpp.mesh.MeshFunctionSizet(mesh_model, mvc_surf)
    t += time.time()
    mpi_print(f"      xdmf read time: {t:0.3f}s")

    return mesh_model, markers, markers_surf
