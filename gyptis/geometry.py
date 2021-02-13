#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


"""
Geometry definition using Gmsh api.
For more information see Gmsh's `documentation <https://gmsh.info/doc/texinfo/gmsh.html>`_
"""


import numbers
import re
import tempfile
from functools import wraps

import gmsh
import numpy as np

from .helpers import Measure
from .mesh import read_mesh
from .plotting import *

geo = gmsh.model.geo
occ = gmsh.model.occ
setnum = gmsh.option.setNumber

gmsh_options = gmsh.option


def _set_opt_gmsh(name, value):
    if isinstance(value, str):
        return gmsh_options.setString(name, value)
    elif isinstance(value, numbers.Number) or isinstance(value, bool):
        if isinstance(value, bool):
            value = int(value)
        return gmsh_options.setNumber(name, value)
    else:
        raise ValueError("value must be string or number")


def _get_opt_gmsh(name):
    try:
        return gmsh_options.getNumber(name)
    except:
        return gmsh_options.getString(name)


setattr(gmsh_options, "set", _set_opt_gmsh)
setattr(gmsh_options, "get", _get_opt_gmsh)


def _add_method(cls, func, name):
    @wraps(func)
    def wrapper(*args, sync=True, **kwargs):
        out = func(*args, **kwargs)
        if sync:
            occ.synchronize()
        return out

    setattr(cls, name, wrapper)
    return func


def _dimtag(tag, dim=3):
    if not isinstance(tag, list):
        tag = list([tag])
    return [(dim, t) for t in tag]


def _get_bnd(id, dim):
    out = gmsh.model.getBoundary(_dimtag(id, dim=dim), False, False, False)
    return [b[1] for b in out]


def _convert_name(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class Geometry(object):
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
        gmsh_args=None,
        kill=False,
        verbose=0,
    ):
        self.model_name = model_name
        self.mesh_name = mesh_name
        self.dim = dim
        self.subdomains = dict(volumes={}, surfaces={}, curves={}, points={})
        self.data_dir = data_dir if data_dir else tempfile.mkdtemp()
        self.occ = occ
        self.mesh_object = {}
        self.measure = {}
        self.mesh = {}
        self.markers = {}

        for object_name in dir(occ):
            if (
                not object_name.startswith("__")
                and object_name != "mesh"
                and object_name not in dir(self)
            ):
                bound_method = getattr(occ, object_name)
                name = _convert_name(bound_method.__name__)
                _add_method(self, bound_method, name)
        self._old_add_ellipse = self.add_ellipse
        del self.add_ellipse
        self._old_add_circle = self.add_circle
        del self.add_circle

        if kill:
            try:
                gmsh.finalize()
            except ValueError:
                pass

        self.gmsh_args = gmsh_args
        if gmsh_args is not None:
            gmsh.initialize(self.gmsh_args)
        else:
            gmsh.initialize()

        gmsh_options.set("General.Verbosity", verbose)

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
        gmsh.model.removePhysicalName(name)
        gmsh.model.setPhysicalName(dim, self.subdomains[dicname][name], name)

    def dimtag(self, id, dim=None):
        """Convert a .

        Parameters
        ----------
        id : int or list of int
            Label or list of labels.
        dim : type
            Dimension.

        Returns
        -------
        int or list of int
            A tuple (dim, tag) or list of such tuples (gmsh DimTag notation).

        """
        dim = dim or self.dim
        return _dimtag(id, dim=dim)

    # def add_circle(self,x, y, z, ax, ay,**kwargs):
    #     ell = self._old_add_ellipse(x, y, z, ax, ay,**kwargs)
    #     ell = self.add_curve_loop([ell])
    #     return self.add_plane_surface([ell])

    def add_circle(self, x, y, z, r, surface=True, **kwargs):
        if surface:
            circ = self._old_add_circle(x, y, z, r, **kwargs)
            circ = self.add_curve_loop([circ])
            return self.add_plane_surface([circ])
        else:
            return self._old_add_circle(x, y, z, r, **kwargs)

    def add_ellipse(self, x, y, z, ax, ay, surface=True, **kwargs):
        if surface:
            ell = self._old_add_ellipse(x, y, z, ax, ay, **kwargs)
            ell = self.add_curve_loop([ell])
            return self.add_plane_surface([ell])
        else:
            return self._old_add_ellipse(x, y, z, ax, ay, **kwargs)

    def fragment(self, id1, id2, dim1=None, dim2=None, sync=True, **kwargs):
        dim1 = dim1 if dim1 else self.dim
        dim2 = dim2 if dim2 else self.dim
        a1 = self.dimtag(id1, dim1)
        a2 = self.dimtag(id2, dim2)
        ov, ovv = occ.fragment(a1, a2, **kwargs)
        if sync:
            occ.synchronize()
        return [o[1] for o in ov]

    def cut(self, id1, id2, dim1=None, dim2=None, sync=True, **kwargs):
        dim1 = dim1 if dim1 else self.dim
        dim2 = dim2 if dim2 else self.dim
        a1 = self.dimtag(id1, dim1)
        a2 = self.dimtag(id2, dim2)
        ov, ovv = occ.cut(a1, a2, **kwargs)
        if sync:
            occ.synchronize()
        return [o[1] for o in ov]

    def fuse(self, id1, id2, dim1=None, dim2=None, sync=True):
        dim1 = dim1 if dim1 else self.dim
        dim2 = dim2 if dim2 else self.dim
        a1 = self.dimtag(id1, dim1)
        a2 = self.dimtag(id2, dim2)
        ov, ovv = occ.fuse(a1, a2)
        if sync:
            occ.synchronize()
        return [o[1] for o in ov]

    def get_boundaries(self, id, dim=None, physical=True):
        dim = dim if dim else self.dim
        if isinstance(id, str):
            if dim == 3:
                type_entity = "volumes"
            elif dim == 2:
                type_entity = "surfaces"
            else:
                type_entity = "curves"
            id = self.subdomains[type_entity][id]

            n = gmsh.model.getEntitiesForPhysicalGroup(dim, id)
            bnds = [_get_bnd(n_, dim=dim) for n_ in n]
            bnds = [item for sublist in bnds for item in sublist]
            return list(dict.fromkeys(bnds))
        else:
            if physical:
                n = gmsh.model.getEntitiesForPhysicalGroup(dim, id)[0]
            else:
                n = id
            return _get_bnd(n, dim=dim)

    # def get_boundaries(self, id, dim=None):
    #     dim = dim if dim else self.dim
    #
    #     n = gmsh.model.getEntitiesForPhysicalGroup(dim, id)[0]
    #     return _get_bnd(n, dim=dim)

    def _set_size(self, id, s, dim=None):
        dim = dim if dim else self.dim
        p = gmsh.model.getBoundary(
            self.dimtag(id, dim=dim), False, False, True
        )  # Get all points
        gmsh.model.mesh.setSize(p, s)

    def _check_subdomains(self):
        groups = gmsh.model.getPhysicalGroups()
        names = [gmsh.model.getPhysicalName(*g) for g in groups]
        for subtype, subitems in self.subdomains.items():
            for id in subitems.copy().keys():
                if id not in names:
                    subitems.pop(id)

    def set_mesh_size(self, params, dim=None):
        dim = dim if dim else self.dim
        if dim == 3:
            type_entity = "volumes"
        elif dim == 2:
            type_entity = "surfaces"
        elif dim == 1:
            type_entity = "curves"
        elif dim == 0:
            type_entity = "points"

        # revert sort so that smaller sizes are set last
        params = dict(
            sorted(params.items(), key=lambda item: float(item[1]), reverse=True)
        )

        for id, p in params.items():
            if isinstance(id, str):
                num = self.subdomains[type_entity][id]
                n = gmsh.model.getEntitiesForPhysicalGroup(dim, num)
                for n_ in n:
                    self._set_size(n_, p, dim=dim)
            else:
                self._set_size(id, p, dim=dim)

    def set_size(self, id, s, dim=None):
        if hasattr(id, "__len__") and not isinstance(id, str):
            for i, id_ in enumerate(id):
                if hasattr(s, "__len__"):
                    s_ = s[i]
                else:
                    s_ = s
                params = {id_: s_}
                self.set_mesh_size(params, dim=dim)
        else:
            self.set_mesh_size({id: s}, dim=dim)

    def read_mesh_info(self):
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
        
        self.mesh = self.mesh_object["mesh"]
        self.markers = self.mesh_object["markers"]

    @property
    def msh_file(self):
        return f"{self.data_dir}/{self.mesh_name}"

    def build(
        self,
        interactive=False,
        generate_mesh=True,
        write_mesh=True,
        read_info=True,
        read_mesh=True,
        finalize=True,
        check_subdomains=True,
    ):
        if check_subdomains:
            self._check_subdomains()

        if generate_mesh:
            self.mesh_object = self.generate_mesh(
                generate=generate_mesh, write=write_mesh, read=read_mesh
            )

        if read_info:
            self.read_mesh_info()

        if interactive:
            gmsh.fltk.run()
        if finalize:
            gmsh.finalize()
        return self.mesh_object

    def read_mesh_file(self):
        return read_mesh(self.msh_file, data_dir=self.data_dir, dim=self.dim)

    def generate_mesh(self, generate=True, write=True, read=True):
        if generate:
            gmsh.model.mesh.generate(self.dim)
        if write:
            gmsh.write(self.msh_file)
        if read:
            return self.read_mesh_file()

    def plot_mesh(self, **kwargs):
        return dolfin.plot(self.mesh,**kwargs)

    def plot_subdomains(self,**kwargs):
        return plot_subdomains(self.markers["triangle"],**kwargs)

class BoxPML2D(Geometry):
    def __init__(
        self,
        box_size=(1, 1),
        box_center=(0, 0),
        pml_width=(0.2, 0.2),
        model_name="2D box with PMLs",
        mesh_name="mesh.msh",
        data_dir=None,
        **kwargs,
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
            return self.add_rectangle(*corner, *rect_size)

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

        self.fragment(self.box, self.pmls)

        self.add_physical([pmlxp, pmlxm], "pmlx")
        self.add_physical([pmlyp, pmlym], "pmly")
        self.add_physical([pmlxypp, pmlxypm, pmlxymm, pmlxymp], "pmlxy")


class BoxPML3D(Geometry):
    def __init__(
        self,
        box_size=(1, 1, 1),
        box_center=(0, 0, 0),
        pml_width=(0.2, 0.2, 0.2),
        model_name="3D box with PMLs",
        mesh_name="mesh.msh",
        data_dir=None,
        **kwargs,
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
            return self.add_box(*corner, *rect_size)

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

        self.fragment(self.box, self.pmls)
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
