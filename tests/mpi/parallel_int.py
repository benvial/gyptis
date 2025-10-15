#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.0
# License: MIT
# See the documentation at gyptis.gitlab.io
from dolfin import MPI

from gyptis import Geometry, dolfin, pi
from gyptis.utils.helpers import mpi_print

rank = MPI.rank(MPI.comm_world)
# dolfin.parameters["ghost_mode"] = "shared_vertex"
dolfin.parameters["ghost_mode"] = "shared_facet"
# dolfin.parameters["ghost_mode"] = "none"
import sys

meshpar = float(sys.argv[1])
square_size = 2
cyl_size = 1
mesh_size = cyl_size / meshpar
model = Geometry("Square", dim=2)
box = model.add_rectangle(
    -square_size / 2, -square_size / 2, 0, square_size, square_size
)
cyl = model.add_rectangle(-cyl_size / 2, -cyl_size / 2, 0, cyl_size, cyl_size)
# cyl = model.add_circle(0, 0, 0, cyl_size)
cyl, box = model.fragment(cyl, box)

model.add_physical(box, "box")
model.add_physical(cyl, "cyl")

# outer_bnds = model.get_boundaries("box")[:-4]
cyl_bnds = model.get_boundaries("cyl")[1]
# model.add_physical(outer_bnds, "outer_bnds", dim=1)
model.add_physical(cyl_bnds, "cyl_bnds", dim=1)
model.set_size("box", 1 * mesh_size)
model.set_size("cyl", 1 * mesh_size)
model.set_size("cyl_bnds", 1 * mesh_size, dim=1)
mesh_object = model.build()


dx = model.measure["dx"]
ds = model.measure["ds"]
dS = model.measure["dS"]

#
# area_cyl = dolfin.assemble(1 * dx("cyl"))
# print(area_cyl)
# assert abs(area_cyl - pi*cyl_size ** 2) < 1e-3
#
#
# out_len = dolfin.assemble(1 * ds("outer_bnds"))
# print(out_len)
# assert abs(out_len - square_size *4) < 1e-3
#
#
# in_len = dolfin.assemble(1 * dS("cyl_bnds"))
# print(in_len)
# assert abs(in_len - cyl_size *2*pi) < 1e-2
#
n_out = model.unit_normal_vector
n = n_out("+")
# in_surf = dolfin.assemble(dolfin.dot(n,n) * dS("cyl_bnds"))
# print(in_surf)
# assert abs(in_surf - cyl_size**2 *4*pi) < 1e-2

in_surf = dolfin.assemble(abs(n_out("+")[0]) * dS("cyl_bnds"))


mpi_print(f" result = {in_surf} (should be 1.0)")


#
#
# model = Geometry("Cube", dim=3)
# box = model.add_box(
#     -square_size / 2, -square_size / 2, -square_size / 2, square_size, square_size,square_size,
# )
# # cyl = model.add_rectangle(-cyl_size / 2, -cyl_size / 2, 0, cyl_size, cyl_size)
# cyl = model.add_sphere(0,0, 0, cyl_size)
# cyl, box = model.fragment(cyl, box)
#
# model.add_physical(box, "box")
# model.add_physical(cyl, "cyl")
#
# outer_bnds = model.get_boundaries("box")[:-1]
# cyl_bnds = model.get_boundaries("cyl")
# model.add_physical(outer_bnds, "outer_bnds", dim=2)
# model.add_physical(cyl_bnds, "cyl_bnds", dim=2)
# model.set_size("box", 1 * mesh_size)
# model.set_size("cyl", 1 * mesh_size)
# model.set_size("cyl_bnds", 1 *mesh_size, dim=2)
# mesh_object = model.build(0)
#
# #
# dx = model.measure["dx"]
# dS = model.measure["dS"]
# ds = model.measure["ds"]
#
# # vol_sphere = dolfin.assemble(1 * dx("cyl"))
# # print(vol_sphere)
# # assert abs(vol_sphere - 4/3*pi*cyl_size ** 3) < 1e-3
# #
# #
# # out_surf = dolfin.assemble(1 * ds("outer_bnds"))
# # print(out_surf)
# # assert abs(out_surf - square_size**2 *6) < 1e-3
# #
# # in_surf = dolfin.assemble(1 * dS("cyl_bnds"))
# # print(in_surf)
# # assert abs(in_surf - cyl_size**2 *4*pi) < 1e-2
# #
# #
#
# n_out = model.unit_normal_vector
# n = n_out("+")
# in_surf = dolfin.assemble(dolfin.dot(n,n) * dS("cyl_bnds"))
# # print(in_surf)
# assert abs(in_surf - cyl_size**2 *4*pi) < 1e-2
#
# in_surf = dolfin.assemble(n_out("+")[0] * dS("cyl_bnds"))
# print(in_surf)
