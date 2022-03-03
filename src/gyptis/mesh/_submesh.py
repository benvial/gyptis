#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

import sys

import gyptis
from gyptis.mesh import read_mesh

meshname = sys.argv[1]
domnum = int(sys.argv[2])
dim = int(sys.argv[3])
outpath = sys.argv[4]

mesh_object = read_mesh(meshname, dim=dim)
mesh = mesh_object["mesh"]
cell = "triangle" if dim == 2 else "tetra"
markers = mesh_object["markers"][cell]


submesh = gyptis.dolfin.SubMesh(mesh, markers, domnum)

with gyptis.dolfin.XDMFFile(submesh.mpi_comm(), outpath) as infile:
    infile.write(submesh)
