#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

import os
import tempfile

import meshio
import numpy as np

from .. import ADJOINT, dolfin


def read_mesh(mesh_file, data_dir=None, data_dir_xdmf=None, dim=3, subdomains=None):
    data_dir_xdmf = data_dir_xdmf or tempfile.mkdtemp()
    meshio_mesh = meshio.read(mesh_file)
    base_cell_type = "tetra" if dim == 3 else "triangle"

    points = meshio_mesh.points[:, :2] if dim == 2 else meshio_mesh.points
    physicals = meshio_mesh.cell_data_dict["gmsh:physical"]

    cell_types, data_gmsh = zip(*physicals.items())
    data_gmsh = list(data_gmsh)
    cells = {ct: [] for ct in cell_types}

    for cell_type in cell_types:
        for cell in meshio_mesh.cells:
            if cell.type == cell_type:
                cells[cell_type].append(cell.data)
        cells[cell_type] = np.vstack(cells[cell_type])

    icell = np.where(np.array(cell_types) == base_cell_type)[0][0]
    if subdomains is not None:
        doms = subdomains if hasattr(subdomains, "__len__") else list([subdomains])
        mask = np.hstack([np.where(data_gmsh[icell] == i) for i in doms])[0]
        data_gmsh_ = data_gmsh[icell][mask]
        data_gmsh[icell] = data_gmsh_
        cells[base_cell_type] = cells[base_cell_type][mask]

    mesh_data = {}

    for cell_type, data in zip(cell_types, data_gmsh):
        meshio_data = meshio.Mesh(
            points=points,
            cells={cell_type: cells[cell_type]},
            cell_data={cell_type: [data]},
        )
        meshio.xdmf.write(f"{data_dir_xdmf}/{cell_type}.xdmf", meshio_data)
        mesh_data[cell_type] = meshio_data

    dolfin_mesh = dolfin.Mesh()
    with dolfin.XDMFFile(
        dolfin_mesh.mpi_comm(), f"{data_dir_xdmf}/{base_cell_type}.xdmf"
    ) as infile:
        infile.read(dolfin_mesh)

    markers = {}
    dim_map = dict(line=1, triangle=2, tetra=3)
    if subdomains is not None:
        cell_types = [base_cell_type]
    for cell_type in cell_types:
        mvc = dolfin.MeshValueCollection("size_t", dolfin_mesh, dim_map[cell_type])
        try:
            with dolfin.XDMFFile(
                dolfin_mesh.mpi_comm(), f"{data_dir_xdmf}/{cell_type}.xdmf"
            ) as infile:
                infile.read(mvc, cell_type)
        except:
            pass

        markers[cell_type] = dolfin.cpp.mesh.MeshFunctionSizet(dolfin_mesh, mvc)
    return dict(mesh=dolfin_mesh, markers=markers)


class MarkedMesh:
    def __init__(self, filename, geometric_dimension=3, data_dir=None):
        self.data_dir = data_dir
        self.geometric_dimension = geometric_dimension
        self.filename = filename
        data_dir = data_dir or tempfile.mkdtemp()
        dic = read_mesh(filename, dim=geometric_dimension, data_dir=data_dir)
        self.mesh = dic["mesh"]
        self.markers = dic["markers"]
        self.dimension = self.mesh.geometric_dimension()


def run_submesh(geom, subdomain, outpath=None):
    outpath = outpath or os.path.join(tempfile.mkdtemp(), "submesh.xdmf")
    fname = os.path.join(os.path.dirname(__file__), "_submesh.py")
    meshname = os.path.join(geom.data_dir, geom.mesh_name)
    domnum = geom.domains[subdomain]
    dim = geom.dim
    os.system(f"mpirun -n 1 python {fname} {meshname} {domnum} {dim} {outpath}")
    return outpath


def read_xdmf_mesh(outpath):
    dolfin_mesh = dolfin.Mesh()
    with dolfin.XDMFFile(dolfin_mesh.mpi_comm(), outpath) as infile:
        infile.read(dolfin_mesh)
    return dolfin_mesh
