#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import tempfile
from distutils.version import LooseVersion

import meshio
import numpy as np
from dolfin import MPI
from fenics_adjoint import __version__ as _fenics_adjoint_version

from . import ADJOINT, dolfin
from .log import *


def read_mesh(mesh_file, data_dir=None, dim=3, subdomains=None):
    meshio_mesh = meshio.read(mesh_file)
    base_cell_type = "tetra" if dim == 3 else "triangle"

    points = meshio_mesh.points[:, :2] if dim == 2 else meshio_mesh.points
    physicals = meshio_mesh.cell_data_dict["gmsh:physical"]

    cell_types, data_gmsh = zip(*physicals.items())
    cells = {ct: [] for ct in cell_types}

    for cell_type in cell_types:
        for cell in meshio_mesh.cells:
            if cell.type == cell_type:
                cells[cell_type].append(cell.data)
        cells[cell_type] = np.vstack(cells[cell_type])

    if subdomains is not None:
        doms = subdomains if hasattr(subdomains, "__len__") else list([subdomains])
        mask = np.hstack([np.where(data_gmsh[0] == i) for i in doms])[0]
        data_gmsh_ = data_gmsh[0][mask]
        data_gmsh = (data_gmsh_,)
        cells[base_cell_type] = cells[base_cell_type][mask]

    mesh_data = {}
    for cell_type, data in zip(cell_types, data_gmsh):
        meshio_data = meshio.Mesh(
            points=points,
            cells={cell_type: cells[cell_type]},
            cell_data={cell_type: [data]},
        )
        meshio.xdmf.write(f"{data_dir}/{cell_type}.xdmf", meshio_data)
        mesh_data[cell_type] = meshio_data

    # avoid bug in dolfin-adjoint: https://github.com/dolfin-adjoint/pyadjoint/issues/38
    if LooseVersion(_fenics_adjoint_version) < LooseVersion("2019.1.1") and ADJOINT:
        _msg = f""" WARNING
        You are running dolfin_adjoint v{_fenics_adjoint_version}.
        Results might be inaccurate when assembling on interior facets in parallel.
        Upgrade using `pip install dolfin_adjoint --upgrade`
        """
        logging.warning(_msg)
        dolfin_mesh = dolfin.Mesh()
        with dolfin.XDMFFile(f"{data_dir}/{base_cell_type}.xdmf") as infile:
            infile.read(dolfin_mesh)
    else:
        dolfin_mesh = dolfin.Mesh(MPI.comm_self)
        with dolfin.XDMFFile(
            MPI.comm_self, f"{data_dir}/{base_cell_type}.xdmf"
        ) as infile:
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
