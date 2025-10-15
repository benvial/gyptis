#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.3
# License: MIT
# See the documentation at gyptis.gitlab.io


import sys

from dolfin import MPI

from gyptis import Geometry, dolfin, pi

comm = MPI.comm_world
rank = MPI.rank(comm)

dolfin.parameters["ghost_mode"] = "shared_vertex"
dolfin.parameters["ghost_mode"] = "shared_facet"


square_size = 1
cyl_size = 0.2
mesh_size = cyl_size / 50

# data_dir = None
data_dir = "/tmp"


def make_geo():
    model = Geometry(
        "Square", dim=2, data_dir=data_dir, options={"General.Verbosity": 4}
    )
    box = model.add_rectangle(
        -square_size / 2, -square_size / 2, 0, square_size, square_size
    )
    # cyl = model.add_rectangle(-cyl_size / 2, -cyl_size / 2, 0, cyl_size, cyl_size)
    cyl = model.add_circle(0, 0, 0, cyl_size)
    cyl, box = model.fragment(cyl, box)

    model.add_physical(box, "box")
    model.add_physical(cyl, "cyl")

    outer_bnds = model.get_boundaries("box")[:-1]
    cyl_bnds = model.get_boundaries("cyl")
    model.add_physical(outer_bnds, "outer_bnds", dim=1)
    model.add_physical(cyl_bnds, "cyl_bnds", dim=1)
    model.set_size("box", 1 * mesh_size)
    model.set_size("cyl", 1 * mesh_size)
    model.set_size("cyl_bnds", 1 * mesh_size, dim=1)
    return model


import time

model = make_geo()

mpi = bool(int(sys.argv[1]))

print("MPI: ", mpi)


if mpi:
    if rank == 0:
        print("meshing")
        sys.stdout.flush()
        model.build(
            interactive=False,
            generate_mesh=True,
            write_mesh=True,
            read_info=False,
            read_mesh=False,
            finalize=True,
            check_subdomains=True,
        )
        data = model.mesh_object
        # model = 3
    else:
        data = None
    # data = comm.bcast(data, root=0)
else:
    model = make_geo()
    model.build()
    data = model.mesh_object

print(data)
data = model.read_mesh_file()
print(data)

# dx = model.measure["dx"]
# ds = model.measure["ds"]
# dS = model.measure["dS"]


#
#
# if rank == 0:
#     print('Process {} computing:'.format(rank))
#     sys.stdout.flush()
#     # model.build()
#     time.sleep(2)
#     print('Process {} done computing:'.format(rank))
#     sys.stdout.flush()
#     data = {'a': 7, 'b': 3.14}
#     comm.send(data, tag=11)
#     print('Process {} sent data:'.format(rank), data)
#     sys.stdout.flush()
# else:
#     print('Process {} waiting:'.format(rank))
#     sys.stdout.flush()
#     data = comm.recv(source=0, tag=11)
#     print('Process {} received data:'.format(rank), data)
#     sys.stdout.flush()

#
# size = 2
#
# if rank == 0:
#     data = {'x': 1, 'y': 2.0}
#     for i in range(1, size):
#         req = comm.isend(data, dest=i, tag=i)
#         req.wait()
#         print('Process {} sent data:'.format(rank), data)
#
# else:
#     req = comm.irecv(source=0, tag=rank)
#     data = req.wait()
#     print('Process {} received data:'.format(rank), data)

#
# mpi = False
# if mpi:
#     if rank ==0:
#         model.build()
#         req = comm.isend(data, dest=1, tag=11)
#     req.wait()
#     else:
#         model.build(
#             generate_mesh=False,
#             write_mesh=False,
#         )
# else:
#     model.build()
#
# dx = model.measure["dx"]
# ds = model.measure["ds"]
# dS = model.measure["dS"]
