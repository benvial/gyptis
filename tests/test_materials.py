#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import pytest
from numpy import e, pi
from test_geometry import geom2D

import gyptis
from gyptis.complex import *
from gyptis.geometry import *
from gyptis.materials import *

model = geom2D(mesh_size=0.01)
mesh = model.mesh_object["mesh"]
dx = model.measure["dx"]
r = model.radius
l = model.square_size

markers = model.mesh_object["markers"]["triangle"]
domains = model.subdomains["surfaces"]


W = dolfin.FunctionSpace(mesh, "CG", 1)


def test_subdomain():
    values = dict(cyl=12, box=1)
    sub = Subdomain(markers, domains, values, degree=1)
    sub_py = Subdomain(markers, domains, values, degree=1, cpp=False)

    a = dolfin.assemble(sub * dx)
    a_py = dolfin.assemble(sub_py * dx)
    assert a == a_py

    a_cyl = pi * r ** 2
    a_box = l ** 2 - a_cyl
    a_test = a_cyl * values["cyl"] + a_box * values["box"]

    assert abs(a - a_test) ** 2 < 1e-6
    #
    # W0 = dolfin.FunctionSpace(mesh, "DG", 0)
    # sub_plot = dolfin.project(sub, W0)
    #
    # import matplotlib.pyplot as plt
    # # s = dolfin.plot(sub_plot)
    # s = dolfin.plot(sub,mesh=mesh)
    # plt.colorbar(s)
    # plt.show()

    f = dolfin.Expression(" exp(-pow(x[0]/r,2) - pow(x[1]/r,2))", degree=0, r=r)

    values = dict(cyl=f, box=1)
    sub_with_function = Subdomain(markers, domains, values, degree=1)
    I = dolfin.assemble(sub_with_function * dx("cyl"))
    Iexact = pi * r ** 2 * (1 - 1 / e)
    assert abs(I - Iexact) ** 2 < 1e-7
    sub_with_function_python = Subdomain(markers, domains, values, degree=1, cpp=False)
    I_python = dolfin.assemble(sub_with_function_python * dx("cyl"))
    assert abs(I_python - Iexact) ** 2 < 1e-7


def test_subdomain_complex():
    values = dict(cyl=12 + 2j, box=1)
    sub = Subdomain(markers, domains, values, degree=1)
    sub_py = Subdomain(
        markers,
        domains,
        values,
        degree=1,
        cpp=False,
    )

    a = assemble(sub * dx)
    a_py = assemble(sub_py * dx)
    assert a == a_py

    a_cyl = pi * r ** 2
    a_box = l ** 2 - a_cyl
    a_test = a_cyl * values["cyl"] + a_box * values["box"]
    assert abs(a - a_test) ** 2 < 1e-6
    f = dolfin.Expression(" exp(-pow(x[0]/r,2) - pow(x[1]/r,2))", degree=0, r=r)

    eps_cyl = 2
    eps_box = f
    mapping = dict(cyl=eps_cyl, box=eps_box)
    eps = Subdomain(markers, domains, mapping, degree=0)
    assert assemble((eps - eps_box) * dx("box")) == 0
    assert assemble((eps - eps_cyl) * dx("cyl")) == 0

    eps_cyl = 2 - 1.2 * 1j
    eps_box = Complex(f, f)
    mapping = dict(cyl=eps_cyl, box=eps_box)
    eps = Subdomain(markers, domains, mapping, degree=0)
    assert assemble((eps - eps_box) * dx("box")) == 0
    assert assemble((eps - eps_cyl) * dx("cyl")) == 0

    eps_cyl = [[1, 2], [3, 4]]
    eps_box = 2
    mapping = dict(cyl=eps_cyl, box=eps_box)
    eps = Subdomain(markers, domains, mapping, degree=0)
    eps_box_tens = eps_box * np.eye(2)
    for i in [0, 1]:
        for j in [0, 1]:
            assert assemble((eps[i][j] - eps_box_tens[i, j]) * dx("box")) == 0
            assert assemble((eps[i][j] - eps_cyl[i][j]) * dx("cyl")) == 0

    eps_cyl = [[1, 2 - 1j], [3, Complex(f, f)]]
    eps_box = 1
    mapping = dict(cyl=eps_cyl, box=eps_box)
    eps = Subdomain(markers, domains, mapping, degree=0)
    eps_box_tens = eps_box * np.eye(2)
    for i in [0, 1]:
        for j in [0, 1]:
            assert assemble((eps[i][j] - eps_box_tens[i, j]) * dx("box")) == 0
            assert assemble((eps[i][j] - eps_cyl[i][j]) * dx("cyl")) == 0


def test_pml():
    pml = PML()

# TODO: be careful here
# Warning
# /ufl/exproperators.py:336: FutureWarning: elementwise comparison failed;
# returning scalar instead, but in the future will perform elementwise comparison
# if arg in ("+", "-"):
# possible solution: check f.ufl_operands and project first
#
# class MySubdomain(dolfin.UserExpression):
#     def __init__(self, markers, val, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.markers=markers
#         self.val =val
#         self.W =W
#
#         if callable(self.val) and self.val.ufl_operands:
#             self.val = dolfin.project(self.val, self.W)
#
#     def eval_cell(self, values, x, cell):
#         if self.markers[cell.index] == 1:
#             if callable(self.val):
#                 values[:] = self.val(x)
#             else:
#                 values[:] = self.val
#         else:
#             values[:] = 1
#
#     def value_shape(self):
#         return ()
#
# import time
#
# f = dolfin.Expression(" exp(-pow(x[0]/r,2) - pow(x[1]/r,2))", degree=1, r=r)
#
#
# values["box"] = 2
# sub = Subdomain(markers, domains, values, degree=1)
# t = -time.time()
# project(sub, W)
# t += time.time()
# print(f"elapsed time: {t:.2f}s")
# #
# values["box"] = f
# sub = Subdomain(markers, domains, values, degree=1)
# t = -time.time()
# project(sub, W)
# t += time.time()
# print(f"elapsed time: {t:.2f}s")
# #
# values["box"] = 2 * f
# sub = Subdomain(markers, domains, values, degree=1)
# t = -time.time()
# project(sub, W)
# t += time.time()
# print(f"elapsed time: {t:.2f}s")
#
# values["box"] = dolfin.project(2 * f, W)
# sub = Subdomain(markers, domains, values, degree=1)
# t = -time.time()
# project(sub, W)
# t += time.time()
# print(f"elapsed time: {t:.2f}s")
