#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from . import dolfin
from .complex import *
from .formulation import Maxwell3D
from .geometry import *
from .materials import *
from .simulation import Simulation
from .source import *


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

        self.add_physical(box, "box")
        self.add_physical(pmlx, "pmlx")
        self.add_physical(pmly, "pmly")
        self.add_physical(pmlz, "pmlz")

        self.add_physical(pmlxy, "pmlxy")
        self.add_physical(pmlyz, "pmlyz")
        self.add_physical(pmlxz, "pmlxz")

        self.add_physical(pml3, "pmlxyz")


class Scatt3D(Simulation):
    def __init__(
        self,
        geometry,
        epsilon,
        mu,
        source,
        boundary_conditions={},
        degree=1,
        mat_degree=1,
        pml_stretch=1 - 1j,
    ):
        assert isinstance(geometry, BoxPML3D)
        assert source.dim == 3
        function_space = ComplexFunctionSpace(geometry.mesh, "N1curl", degree)
        pmls = []
        pml_names = []
        for direction in ["x", "y", "z", "xy", "yz", "xz", "xyz"]:
            pml_name = f"pml{direction}"
            pml_names.append(pml_name)
            pmls.append(
                PML(
                    direction,
                    stretch=pml_stretch,
                    matched_domain="box",
                    applied_domain=pml_name,
                )
            )

        epsilon_coeff = Coefficient(
            epsilon,
            geometry,
            pmls=pmls,
            degree=mat_degree,
            dim=3,
        )
        mu_coeff = Coefficient(
            mu,
            geometry,
            pmls=pmls,
            degree=mat_degree,
            dim=3,
        )

        coefficients = epsilon_coeff, mu_coeff
        no_source_domains = ["box"] + pml_names
        source_domains = [
            dom for dom in geometry.domains if dom not in no_source_domains
        ]

        formulation = Maxwell3D(
            geometry,
            coefficients,
            function_space,
            source=source,
            source_domains=source_domains,
            reference="box",
            boundary_conditions=boundary_conditions,
        )

        super().__init__(geometry, formulation)

    def solve_system(self, again=False):
        E = super().solve_system(again=again, vector_function=False)
        self.solution = {}
        self.solution["diffracted"] = E
        self.solution["total"] = E + self.source.expression
        return E
