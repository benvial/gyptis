#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.2
# License: MIT
# See the documentation at gyptis.gitlab.io


from .geometry import *
from .geometry import _newer_gmsh


class BoxPML2D(Geometry):
    def __init__(
        self,
        box_size=(1, 1),
        box_center=(0, 0),
        pml_width=(0.2, 0.2),
        Rcalc=0,
        **kwargs,
    ):
        super().__init__(
            dim=2,
            **kwargs,
        )
        self.box_size = box_size
        self.box_center = box_center
        self.pml_width = pml_width
        self.Rcalc = Rcalc

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

        self.pml_physical = [
            "pmlx",
            "pmly",
            "pmlxy",
        ]

        if Rcalc > 0:
            cyl_calc = self.add_circle(*self.box_center, 0, Rcalc)
            box, cyl_calc = self.fragment(box, cyl_calc)
            self.box = [box, cyl_calc]

        self.add_physical(box, "box")
        self.add_physical([pmlxp, pmlxm], "pmlx")
        self.add_physical([pmlyp, pmlym], "pmly")
        self.add_physical([pmlxypp, pmlxypm, pmlxymm, pmlxymp], "pmlxy")

        if Rcalc > 0:
            bnds = self.get_boundaries("box")
            self.calc_bnds = bnds[0]
            self.add_physical(self.calc_bnds, "calc_bnds", dim=1)

    def build(self, *args, **kwargs):
        if _newer_gmsh:
            if self.Rcalc > 0:
                bnds = self.get_boundaries("box")
                self.calc_bnds = bnds[-2]
                self.add_physical(self.calc_bnds, "calc_bnds", dim=1)
        super().build(*args, **kwargs)


class LayeredBoxPML2D(Geometry):
    def __init__(
        self,
        width,
        thicknesses,
        pml_width,
        **kwargs,
    ):
        super().__init__(
            dim=2,
            **kwargs,
        )

        self.width = width
        self.thicknesses = thicknesses
        self.total_thickness = sum(self.thicknesses.values())

        self.layers = {}
        self.y_position = {}
        self.box_size = width, self.total_thickness
        self.pml_width = pml_width

        def _addrect_center(rect_size):
            corner = -np.array(rect_size) / 2
            corner = tuple(corner) + (0,)
            return self.add_rectangle(*corner, *rect_size)

        def _translate(tag, t):
            translation = tuple(t) + (0,)
            self.translate(self.dimtag(tag), *translation)

        def _add_pml(rect_size, translation):
            pml = _addrect_center(rect_size)
            _translate(pml, translation)
            return pml

        y0 = -self.total_thickness / 2
        tx0 = self.pml_width[0] / 2 + self.box_size[0] / 2

        for name, thickness in self.thicknesses.items():
            layer = self.make_layer(y0, thickness)
            self.layers[name] = layer
            self.y_position[name] = y0
            self.add_physical(layer, name)
            sx = (self.pml_width[0], thickness)
            ty0 = thickness / 2 + y0
            tx = np.array([tx0, ty0])
            pmlxp = _add_pml(sx, tx)

            tx = np.array([-tx0, ty0])
            pmlxm = _add_pml(sx, tx)
            self.add_physical([pmlxp, pmlxm], "pmlx_" + name)

            y0 += thickness

        sy = (self.box_size[0], self.pml_width[1])
        ty = np.array([0, self.pml_width[1] / 2 + self.box_size[1] / 2])

        pmlyp = _add_pml(sy, ty)
        pmlym = _add_pml(sy, -ty)

        names = list(self.layers.keys())
        self.add_physical(pmlym, "pmly_" + names[0])
        self.add_physical(pmlyp, "pmly_" + names[-1])

        s = (self.pml_width[0], self.pml_width[1])
        t = np.array(
            [
                self.pml_width[0] / 2 + self.box_size[0] / 2,
                self.pml_width[1] / 2 + self.box_size[1] / 2,
            ]
        )
        pmlxypp = _add_pml(s, t)
        pmlxymm = _add_pml(s, -t)
        pmlxymp = _add_pml(s, (-t[0], t[1]))
        pmlxypm = _add_pml(s, (t[0], -t[1]))

        self.add_physical([pmlxypm, pmlxymm], "pmlxy_" + names[0])
        self.add_physical([pmlxypp, pmlxymp], "pmlxy_" + names[-1])

        self.pmls = []
        for name in names:
            self.pmls.append("pmlx_" + name)
        for name in [names[-1], names[0]]:
            self.pmls.append("pmly_" + name)
            self.pmls.append("pmlxy_" + name)

        self.remove_all_duplicates()

        for sub, num in self.subdomains_entities["surfaces"].items():
            self.add_physical(num, sub)

        x0 = self.width / 2 + self.pml_width[0]
        y0 = (
            list(self.y_position.values())[-1]
            + list(self.thicknesses.values())[-1]
            + self.pml_width[1]
        )
        y1 = list(self.y_position.values())[0] - self.pml_width[1]

        horizontal_lines = self.find_lines_near_positions("x", [-x0, x0])
        vertical_lines = self.find_lines_near_positions("y", [y0, y1])
        outer_bnds = horizontal_lines + vertical_lines
        self.add_physical(outer_bnds, "outer_boundaries", dim=1)

    def make_layer(self, y_position, thickness):
        return self.add_rectangle(-self.width / 2, y_position, 0, self.width, thickness)

    def find_lines_near_positions(self, axis, positions, tol_dir=1e-12, tol_pos=1e-12):
        """
        Generic helper for finding axis-aligned lines near given coordinates.

        axis = "x"  -> find horizontal lines near x = x0
        axis = "y"  -> find vertical   lines near y = y0
        """
        assert axis in ("x", "y"), "axis must be 'x' or 'y'"
        results = []

        for dim, tag in self.get_entities(dim=1):
            nodes = self.get_boundaries(tag, dim=1, physical=False)
            pts = [self.model.get_value(0, n, []) for n in nodes]
            if len(pts) != 2:
                continue
            (x1, y1, _), (x2, y2, _) = pts

            if axis == "y":
                # horizontal lines near x = x0
                for y0 in positions:
                    if y1 == y2 == y0:
                        results.append(tag)

            else:
                # horizontal lines near x = x0
                for x0 in positions:
                    if x1 == x2 == x0:
                        results.append(tag)
        return results
