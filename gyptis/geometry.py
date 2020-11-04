#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import os
import tempfile
import time
from functools import wraps  # This convenience func preserves name and docstring

import gmsh
import numpy as np

from gyptis.helpers import Measure
from gyptis.mesh import read_mesh


def _add_method(cls, func, name):
    @wraps(func)
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        occ.synchronize()
        return out

    setattr(cls, name, wrapper)
    # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
    return func  # returning func means func can still be used normally


occ = gmsh.model.occ
setnum = gmsh.option.setNumber


def _dimtag(tag, dim=3):
    if not isinstance(tag, list):
        tag = list([tag])
    return [(dim, t) for t in tag]


def remove(id, dim=3):
    occ.remove(_dimtag(id, dim=3), True)


def _get_bnd(id, dim):
    out = gmsh.model.getBoundary(_dimtag(id, dim=dim), False, False, False)
    return [b[1] for b in out]


def make_group(dic, dim=3):
    subdomain_dict = {}
    for id, tag in dic.items():
        if not isinstance(tag, list):
            tag = list([tag])
        subdomain_dict[id] = gmsh.model.addPhysicalGroup(dim, tag)
        gmsh.model.setPhysicalName(dim, subdomain_dict[id], id)
    return subdomain_dict


class Model(object):
    """Base class for geometry models.

    Parameters
    ----------
    model_name : str
        Name of the model
    mesh_name : str
        Name of the mesh file (.msh).
    data_dir : str
        Directory to store generated mesh and data.
    dim : int
        Dimension of the problem.

    Attributes
    ----------
    subdomains : dict
        Dictionary containing mapping from physical domains names to their
        index.
    model_name
    mesh_name
    dim
    data_dir

    """

    def __init__(
        self,
        model_name="geometry",
        mesh_name="mesh.msh",
        data_dir=None,
        dim=3,
        kill=False,
    ):
        self.model_name = model_name
        self.mesh_name = mesh_name
        self.dim = dim
        self.subdomains = dict(volumes={}, surfaces={}, curves={}, points={})
        self.data_dir = data_dir if data_dir else tempfile.mkdtemp()

        for object_name in dir(occ):
            if not object_name.startswith("__") and object_name != "mesh":
                bound_method = getattr(occ, object_name)
                name = bound_method.__name__
                _add_method(self, bound_method, name)

        if kill:
            try:
                gmsh.finalize()
            except ValueError:
                pass
        gmsh.initialize(["-format", "msh2"])

    def add_physical(self, id, name, dim=None):
        """Add a physical domain.

        Parameters
        ----------
        id : int or list of int
            The identifiant(s) of elementary entities makling the physical domain.
        name : str
            Name of the domain.
        dim : int
            Dimension.
        """
        dim = dim if dim else self.dim
        dicname = list(self.subdomains)[3 - dim]
        if not isinstance(id, list):
            id = list([id])
        self.subdomains[dicname][name] = gmsh.model.addPhysicalGroup(dim, id)
        gmsh.model.setPhysicalName(dim, self.subdomains[dicname][name], name)

    def dimtag(self, id, dim=None):
        dim = dim or self.dim
        return _dimtag(id, dim=dim)

    def fragmentize(self, id1, id2, dim1=None, dim2=None, sync=True):
        dim1 = dim1 if dim1 else self.dim
        dim2 = dim2 if dim2 else self.dim
        a1 = self.dimtag(id1, dim1)
        a2 = self.dimtag(id2, dim2)
        ov, ovv = occ.fragment(a1, a2)
        if sync:
            occ.synchronize()
        return [o[1] for o in ov]

    def join(self, id1, id2, dim1=None, dim2=None, sync=True):
        dim1 = dim1 if dim1 else self.dim
        dim2 = dim2 if dim2 else self.dim
        a1 = self.dimtag(id1, dim1)
        a2 = self.dimtag(id2, dim2)
        ov, ovv = occ.fuse(a1, a2)
        if sync:
            occ.synchronize()
        return [o[1] for o in ov]

    def get_boundaries(self, id, dim=None):
        dim = dim if dim else self.dim
        return _get_bnd(id, dim=dim)

    def set_size(self, id, s, dim=None):
        dim = dim if dim else self.dim
        p = gmsh.model.getBoundary(
            self.dimtag(id, dim=dim), False, False, True
        )  # Get all points
        gmsh.model.mesh.setSize(p, s)

    def build(
        self, interactive=False, generate_mesh=True, write_mesh=True, finalize=True
    ):
        self.mesh_object = {}
        if generate_mesh:
            self.mesh_object = self._mesh(generate=generate_mesh, write=write_mesh)

            self.measure = {}
            if self.dim == 2:
                marker_dim = "triangle"
                sub_dim = "surfaces"
                marker_dim_minus_1 = "line"
                sub_dim_dim_minus_1 = "curves"
            else:
                marker_dim = "tetra"
                sub_dim = "volumes"
                marker_dim_minus_1 = "triangle"
                sub_dim_dim_minus_1 = "surfaces"

            self.measure["dx"] = Measure(
                "dx",
                domain=self.mesh_object["mesh"],
                subdomain_data=self.mesh_object["markers"][marker_dim],
                subdomain_dict=self.subdomains[sub_dim],
            )

            ## exterior_facets
            if (marker_dim_minus_1 in self.mesh_object["markers"].keys()) and (
                sub_dim_dim_minus_1 in self.subdomains.keys()
            ):
                self.measure["ds"] = Measure(
                    "ds",
                    domain=self.mesh_object["mesh"],
                    subdomain_data=self.mesh_object["markers"][marker_dim_minus_1],
                    subdomain_dict=self.subdomains[sub_dim_dim_minus_1],
                )

                ## interior_facets

                self.measure["dS"] = Measure(
                    "dS",
                    domain=self.mesh_object["mesh"],
                    subdomain_data=self.mesh_object["markers"][marker_dim_minus_1],
                    subdomain_dict=self.subdomains[sub_dim_dim_minus_1],
                )

        if interactive:
            gmsh.fltk.run()
        if finalize:
            gmsh.finalize()
        return self.mesh_object

    def _mesh(self, generate=True, write=True):
        if generate:
            gmsh.model.mesh.generate(self.dim)
        if write:
            msh = f"{self.data_dir}/{self.mesh_name}"
            gmsh.write(msh)
            return read_mesh(msh, data_dir=self.data_dir, dim=self.dim)


class BoxPML2D(Model):
    def __init__(
        self,
        box_size=(1, 1),
        box_center=(0, 0),
        pml_width=(0.2, 0.2),
        model_name="2D box with PMLs",
        mesh_name="mesh.msh",
        data_dir=None,
    ):
        super().__init__(
            model_name=model_name, mesh_name=mesh_name, data_dir=data_dir, dim=2
        )
        self.box_size = box_size
        self.box_center = box_center
        self.pml_width = pml_width

        def _addrect_center(rect_size):
            corner = -np.array(rect_size) / 2
            corner = tuple(corner) + (0,)
            return self.addRectangle(*corner, *rect_size)

        def _translate(tag, t):
            translation = tuple(t) + (0,)
            self.translate(self.dimtag(tag), *translation)

        def _add_pml(s, t):
            pml = _addrect_center(s)
            _translate(pml, t)
            return pml

        box = _addrect_center(self.box_size)
        s = (self.pml_width[0], self.box_size[1])
        t = np.array([self.pml_width[0] / 2 + self.box_size[0] / 2, 0])
        pmlxp = _add_pml(s, t)
        pmlxm = _add_pml(s, -t)
        s = (self.box_size[0], self.pml_width[1])
        t = np.array([0, self.pml_width[1] / 2 + self.box_size[1] / 2])
        pmlyp = _add_pml(s, t)
        pmlym = _add_pml(s, -t)

        s = (self.pml_width[0], self.pml_width[1])
        t = np.array(
            [
                self.pml_width[0] / 2 + self.box_size[0] / 2,
                self.pml_width[1] / 2 + self.box_size[1] / 2,
            ]
        )
        pmlxypp = _add_pml(s, t)
        pmlxymm = _add_pml(s, -t)
        pmlxypm = _add_pml(s, (-t[0], t[1]))
        pmlxymp = _add_pml(s, (t[0], -t[1]))

        all_dom = [
            box,
            pmlxp,
            pmlxm,
            pmlyp,
            pmlym,
            pmlxypp,
            pmlxypm,
            pmlxymm,
            pmlxymp,
        ]
        _translate(all_dom, self.box_center)

        self.box = box
        self.pmls = all_dom[1:]

        self.add_physical([pmlxp, pmlxm], "pmlx")
        self.add_physical([pmlyp, pmlym], "pmly")
        self.add_physical([pmlxypp, pmlxypm, pmlxymm, pmlxymp], "pmlxy")


class BoxPML3D(Model):
    def __init__(
        self,
        box_size=(1, 1, 1),
        box_center=(0, 0, 0),
        pml_width=(0.2, 0.2, 0.2),
        model_name="3D box with PMLs",
        mesh_name="mesh.msh",
        data_dir=None,
    ):
        super().__init__(
            model_name=model_name, mesh_name=mesh_name, data_dir=data_dir, dim=3
        )
        self.box_size = box_size
        self.box_center = box_center
        self.pml_width = pml_width

        def _addbox_center(rect_size):
            corner = -np.array(rect_size) / 2
            corner = tuple(corner)
            return self.addBox(*corner, *rect_size)

        def _translate(tag, t):
            translation = tuple(t)
            self.translate(self.dimtag(tag), *translation)

        def _add_pml(s, t):
            pml = _addbox_center(s)
            _translate(pml, t)
            return pml

        box = _addbox_center(self.box_size)
        T = np.array(self.pml_width) / 2 + np.array(self.box_size) / 2

        s = (self.pml_width[0], self.box_size[1], self.box_size[2])
        t = np.array([T[0], 0, 0])
        pmlxp = _add_pml(s, t)
        pmlxm = _add_pml(s, -t)
        s = (self.box_size[0], self.pml_width[1], self.box_size[2])
        t = np.array([0, T[1], 0])
        pmlyp = _add_pml(s, t)
        pmlym = _add_pml(s, -t)
        s = (self.box_size[0], self.box_size[1], self.pml_width[2])
        t = np.array([0, 0, T[2]])
        pmlzp = _add_pml(s, t)
        pmlzm = _add_pml(s, -t)

        s = (self.pml_width[0], self.pml_width[1], self.box_size[2])

        pmlxypp = _add_pml(s, [T[0], T[1], 0])
        pmlxypm = _add_pml(s, [T[0], -T[1], 0])
        pmlxymp = _add_pml(s, [-T[0], T[1], 0])
        pmlxymm = _add_pml(s, [-T[0], -T[1], 0])

        s = (self.box_size[0], self.pml_width[1], self.pml_width[2])
        pmlyzpp = _add_pml(s, [0, T[1], T[0]])
        pmlyzpm = _add_pml(s, [0, T[1], -T[0]])
        pmlyzmp = _add_pml(s, [0, -T[1], T[0]])
        pmlyzmm = _add_pml(s, [0, -T[1], -T[0]])

        s = (self.pml_width[0], self.box_size[1], self.pml_width[2])
        pmlxzpp = _add_pml(s, [T[0], 0, T[2]])
        pmlxzpm = _add_pml(s, [T[0], 0, -T[2]])
        pmlxzmp = _add_pml(s, [-T[0], 0, T[2]])
        pmlxzmm = _add_pml(s, [-T[0], 0, -T[2]])

        s = (self.pml_width[0], self.pml_width[1], self.pml_width[2])
        pmlxyzppp = _add_pml(s, [T[0], T[1], T[2]])
        pmlxyzppm = _add_pml(s, [T[0], T[1], -T[2]])
        pmlxyzpmp = _add_pml(s, [T[0], -T[1], T[2]])
        pmlxyzpmm = _add_pml(s, [T[0], -T[1], -T[2]])
        pmlxyzmpp = _add_pml(s, [-T[0], T[1], T[2]])
        pmlxyzmpm = _add_pml(s, [-T[0], T[1], -T[2]])
        pmlxyzmmp = _add_pml(s, [-T[0], -T[1], T[2]])
        pmlxyzmmm = _add_pml(s, [-T[0], -T[1], -T[2]])

        pmlx = [pmlxp, pmlxm]
        pmly = [pmlyp, pmlym]
        pmlz = [pmlzp, pmlzm]
        pml1 = pmlx + pmly + pmlz

        pmlxy = [pmlxypp, pmlxypm, pmlxymp, pmlxymm]
        pmlyz = [pmlyzpp, pmlyzpm, pmlyzmp, pmlyzmm]
        pmlxz = [pmlxzpp, pmlxzpm, pmlxzmp, pmlxzmm]
        pml2 = pmlxy + pmlyz + pmlxz

        pml3 = [
            pmlxyzppp,
            pmlxyzppm,
            pmlxyzpmp,
            pmlxyzpmm,
            pmlxyzmpp,
            pmlxyzmpm,
            pmlxyzmmp,
            pmlxyzmmm,
        ]

        self.box = box
        self.pmls = pml1 + pml2 + pml3

        _translate([self.box] + self.pmls, self.box_center)
        #

        self.add_physical(pmlx, "pmlx")
        self.add_physical(pmly, "pmly")
        self.add_physical(pmlz, "pmlz")

        self.add_physical(pmlxy, "pmlxy")
        self.add_physical(pmlyz, "pmlyz")
        self.add_physical(pmlxz, "pmlxz")

        self.add_physical(pml3, "pmlxyz")


class BoxPML(object):
    def __new__(self, dim=3, *args, **kwargs):
        if dim not in [2, 3]:
            raise ValueError("dimension must be 2 or 3")
        if dim == 3:
            return BoxPML3D(*args, **kwargs)
        else:
            return BoxPML2D(*args, **kwargs)


if __name__ == "__main__":

    # cds
    #
    # model = BoxPML3D()
    # sphere = model.addSphere(0, 0, 0, 0.1)
    # sphere, model.box = model.fragmentize(sphere, model.box)
    # model.add_physical(sphere, "sphere")
    # model.add_physical(model.box, "box")
    # model.set_size(model.box, 0.06)
    # model.set_size(model.pmls, 0.1)
    # model.set_size(sphere, 0.04)
    # mesh_object = model.build(interactive=True, generate_mesh=True)
    #
    #
    # csc

    model = BoxPML(2)
    cyl = model.addDisk(0, 0, 0, 0.1, 0.1)
    cyl, model.box = model.fragmentize(cyl, model.box)
    model.add_physical(cyl, "cyl")
    model.add_physical(model.box, "box")
    model.set_size(model.box, 0.03)
    model.set_size(model.pmls, 0.1)
    model.set_size(cyl, 0.01)

    # box = model.addRectangle(-1, -1, 0, 2, 2)
    mesh_object = model.build(interactive=True, generate_mesh=True)

    xsa

    #
    # 2D
    model = Model("Scattering from a cylinder", dim=2)
    box = model.addRectangle(-1, -1, 0, 2, 2)
    outer_bnds = model.get_boundaries(box)
    cyl = model.addDisk(0, 0, 0, 0.5, 0.5)

    rect = model.addRectangle(0, 0, 0, 0.8, 0.8)
    cyl = model.join(cyl, rect)

    cyl, box = model.fragmentize(cyl, box)

    model.set_size(box, 0.1)
    model.set_size(cyl, 0.04)

    model.add_physical(cyl, "cyl")
    model.add_physical(box, "box")
    model.add_physical(outer_bnds, "outer_bnds", dim=1)
    print(model.subdomains)
    mesh_object = model.build(interactive=True)

    # import dolfin as df
    #
    # df.File(f"mesh.pvd") << mesh_object["mesh"]
    # df.File(f"markers_bnd.pvd") << mesh_object["markers"]["line"]
    # df.File(f"markers.pvd") << mesh_object["markers"]["triangle"]
    #
    #
    # # 3D
    # model = Model("Scattering from a sphere")
    # box = model.addBox(-1, -1, -1, 2, 2, 2)
    # sphere = model.addSphere(0, 0, 0, 0.5)
    # sphere,box = model.fragmentize(sphere,box)
    # model.set_size(box, 0.1)
    # model.set_size(sphere, 0.1)
    # outer_bnds = model.get_boundaries(box)
    #
    # model.add_physical(sphere, "sphere")
    # model.add_physical(box, "box")
    # model.add_physical(outer_bnds, "outer_bnds",dim=2)
    # print(model.subdomains)
    # model.mesh()
    # model.build(interactive=True)
