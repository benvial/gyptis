#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import dolfin as df
import meshio
import numpy as np


def read_mesh(mesh_file, data_dir="./data", dim=3):
    msh = meshio.read(mesh_file)

    # celltypes = set([c.type for c in msh.cells])

    celltypes = msh.cell_data_dict["gmsh:physical"].keys()

    data_gmsh = msh.cell_data_dict["gmsh:physical"]

    ## 2d
    cells = {}
    data = {}
    for c in celltypes:
        cells[c] = []
    for c in celltypes:
        for cell in msh.cells:
            if cell.type == c:
                cells[c].append(cell.data)
        cells[c] = np.vstack(cells[c])

    points = msh.points[:, :2] if dim == 2 else msh.points

    _mesh_data = {}
    for c in celltypes:
        # meshio.Mesh(points=points, cells=cells, cell_data=data_gmsh)
        _mesh_data[c] = meshio.Mesh(
            points=points,
            cells={c: cells[c]},
            cell_data={c: [data_gmsh[c]]},
        )

        # meshio.write(f"{data_dir}/mesh.xdmf", _mesh_data["line"])
        filename = f"{data_dir}/{c}.xdmf"
        meshio.xdmf.write(filename, _mesh_data[c])

    ### markers
    mesh_object = {}
    mesh_model = df.Mesh()

    c = "tetra" if dim == 3 else "triangle"
    filename = f"{data_dir}/{c}.xdmf"
    with df.XDMFFile(filename) as infile:
        infile.read(mesh_model)

    mesh_object["mesh"] = mesh_model

    markers = {}

    dim_map = dict(line=1, triangle=2, tetra=3)
    for c in celltypes:
        filename = f"{data_dir}/{c}.xdmf"
        i = dim_map[c]
        mvc = df.MeshValueCollection("size_t", mesh_model, i)
        with df.XDMFFile(filename) as infile:
            infile.read(mvc, c)
        markers[c] = df.cpp.mesh.MeshFunctionSizet(mesh_model, mvc)

    mesh_object["markers"] = markers
    # mesh_object["markers"]= df.cpp.mesh.MeshFunctionSizet(mesh_model, mvc_surf)
    # mesh_object["markers"]= df.cpp.mesh.MeshFunctionSizet(mesh_model, mvc_surf)

    return mesh_object
