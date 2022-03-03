#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


from .geometry import *


class Layered2D(Geometry):
    def __init__(
        self,
        period=1,
        thicknesses=OrderedDict(),
        **kwargs,
    ):
        super().__init__(
            dim=2,
            **kwargs,
        )
        self.period = period
        # assert isinstance(self.thicknesses == OrderedDict)
        self.thicknesses = thicknesses

        self.layers = list(thicknesses.keys())

        self.total_thickness = sum(self.thicknesses.values())

        self.y0 = -sum(list(self.thicknesses.values())[:2])
        self.layers = {}
        self.y_position = {}
        y0 = self.y0
        self._phys_groups = []
        for id, thickness in self.thicknesses.items():
            layer = self.make_layer(y0, thickness)
            self.layers[id] = layer
            self.y_position[id] = y0
            self.add_physical(layer, id)
            self._phys_groups.append(layer)
            y0 += thickness

        self.remove_all_duplicates()
        self.synchronize()

        for sub, num in self.subdomains["surfaces"].items():
            self.add_physical(num, sub)

    @property
    def translation_x(self):
        return self._translation_matrix([self.period, 0, 0])

    def make_layer(self, y_position, thickness):
        box = self.add_rectangle(
            -self.period / 2, y_position, 0, self.period, thickness
        )
        return box

    def build(self, *args, **kwargs):

        s = self.get_periodic_bnds(self.y0, self.total_thickness)
        periodic_id = {}
        for k, v in s.items():
            periodic_id[k] = [S[-1] for S in v]
        gmsh.model.mesh.setPeriodic(
            1, periodic_id["+x"], periodic_id["-x"], self.translation_x
        )
        super().build(*args, **kwargs)

    def get_periodic_bnds(self, y_position, thickness, eps=1e-3):
        s = {}

        pmin = -self.period / 2 - eps, -eps + y_position, -eps
        pmax = -self.period / 2 + eps, y_position + thickness + eps, eps
        s["-x"] = gmsh.model.getEntitiesInBoundingBox(*pmin, *pmax, 1)

        pmin = +self.period / 2 - eps, -eps + y_position, -eps
        pmax = +self.period / 2 + eps, y_position + thickness + eps, eps
        s["+x"] = gmsh.model.getEntitiesInBoundingBox(*pmin, *pmax, 1)

        return s
