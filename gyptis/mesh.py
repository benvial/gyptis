#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import tempfile

import meshio
import numpy as np

from . import dolfin


def read_mesh(mesh_file, data_dir=None, dim=3):
    meshio_mesh = meshio.read(mesh_file)

    points = meshio_mesh.points[:, :2] if dim == 2 else meshio_mesh.points
    physicals = meshio_mesh.cell_data_dict["gmsh:physical"]

    cell_types, data_gmsh = zip(*physicals.items())
    cells = {ct: [] for ct in cell_types}

    for cell_type in cell_types:
        for cell in meshio_mesh.cells:
            if cell.type == cell_type:
                cells[cell_type].append(cell.data)
        cells[cell_type] = np.vstack(cells[cell_type])

    mesh_data = {}
    for cell_type, data in zip(cell_types, data_gmsh):
        meshio_data = meshio.Mesh(
            points=points,
            cells={cell_type: cells[cell_type]},
            cell_data={cell_type: [data]},
        )
        meshio.xdmf.write(f"{data_dir}/{cell_type}.xdmf", meshio_data)
        mesh_data[cell_type] = meshio_data

    dolfin_mesh = dolfin.Mesh()

    cell_type = "tetra" if dim == 3 else "triangle"
    with dolfin.XDMFFile(f"{data_dir}/{cell_type}.xdmf") as infile:
        infile.read(dolfin_mesh)

    markers = {}

    dim_map = dict(line=1, triangle=2, tetra=3)
    for cell_type in cell_types:
        mvc = dolfin.MeshValueCollection("size_t", dolfin_mesh, dim_map[cell_type])
        with dolfin.XDMFFile(f"{data_dir}/{cell_type}.xdmf") as infile:
            infile.read(mvc, cell_type)
        markers[cell_type] = dolfin.cpp.mesh.MeshFunctionSizet(dolfin_mesh, mvc)

    return dict(mesh=dolfin_mesh, markers=markers)


class MarkedMesh(object):
    def __init__(self, filename, geometric_dimension=3, data_dir=None):
        self.data_dir = data_dir
        self.geometric_dimension = geometric_dimension
        self.filename = filename

        data_dir = data_dir or tempfile.mkdtemp()
        dic = read_mesh(filename, dim=geometric_dimension, data_dir=data_dir)
        self.mesh = dic["mesh"]
        self.markers = dic["markers"]
        self.dimension = self.mesh.geometric_dimension()
