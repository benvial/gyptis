#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.2
# License: MIT
# See the documentation at gyptis.gitlab.io


from dolfin import *

# parameters['allow_extrapolation'] = True

mesh = UnitIntervalMesh(15)
V = FunctionSpace(mesh, "CG", 2)

u = Function(V)
# u_vec = u.vector()
# u_vec[:] = MPI.comm_world.rank + 1
#
# v_vec = Vector(MPI.comm_self, u_vec.local_size())
# u_vec.gather(v_vec, V.dofmap().dofs())

f = interpolate(Expression("x[0]", degree=2), V)
#


mpi_comm = u.function_space().mesh().mpi_comm()
array = u.vector().get_local()

# gather solution from all processes on proc 0
array_gathered = mpi_comm.gather(array, root=0)

# compute coefficients on proc 0
if mpi_comm.Get_rank() == 0:
    print(array_gathered)
else:
    print(array)
    coef = None

# broadcast from proc 0 to other processes
# mpi_comm.Bcast(coef, root=0)

# print(f(0))
# print(f(1))

#
# v_vec(0)


# print("Original vs copied: ", u_vec.get_local(), v_vec.get_local())  # [1, 1], [0, 0, 0]
