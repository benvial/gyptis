#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

from .geometry import *


class BoxPML3D(Geometry):
    def __init__(
        self,
        box_size=(1, 1, 1),
        box_center=(0, 0, 0),
        pml_width=(0.2, 0.2, 0.2),
        Rcalc=0,
        **kwargs,
    ):
        super().__init__(
            dim=3,
            **kwargs,
        )
        self.box_size = box_size
        self.box_center = box_center
        self.pml_width = pml_width
        self.Rcalc = Rcalc

        box = self._addbox_center(self.box_size)
        T = np.array(self.pml_width) / 2 + np.array(self.box_size) / 2

        s = (self.pml_width[0], self.box_size[1], self.box_size[2])
        t = np.array([T[0], 0, 0])
        pmlxp = self._add_pml(s, t)
        pmlxm = self._add_pml(s, -t)

        s = (self.box_size[0], self.pml_width[1], self.box_size[2])
        t = np.array([0, T[1], 0])
        pmlyp = self._add_pml(s, t)
        pmlym = self._add_pml(s, -t)

        s = (self.box_size[0], self.box_size[1], self.pml_width[2])
        t = np.array([0, 0, T[2]])
        pmlzp = self._add_pml(s, t)
        pmlzm = self._add_pml(s, -t)

        s = (self.pml_width[0], self.pml_width[1], self.box_size[2])
        pmlxypp = self._add_pml(s, [T[0], T[1], 0])
        pmlxypm = self._add_pml(s, [T[0], -T[1], 0])
        pmlxymp = self._add_pml(s, [-T[0], T[1], 0])
        pmlxymm = self._add_pml(s, [-T[0], -T[1], 0])

        s = (self.box_size[0], self.pml_width[1], self.pml_width[2])
        pmlyzpp = self._add_pml(s, [0, T[1], T[2]])
        pmlyzpm = self._add_pml(s, [0, T[1], -T[2]])
        pmlyzmp = self._add_pml(s, [0, -T[1], T[2]])
        pmlyzmm = self._add_pml(s, [0, -T[1], -T[2]])

        s = (self.pml_width[0], self.box_size[1], self.pml_width[2])
        pmlxzpp = self._add_pml(s, [T[0], 0, T[2]])
        pmlxzpm = self._add_pml(s, [T[0], 0, -T[2]])
        pmlxzmp = self._add_pml(s, [-T[0], 0, T[2]])
        pmlxzmm = self._add_pml(s, [-T[0], 0, -T[2]])

        s = (self.pml_width[0], self.pml_width[1], self.pml_width[2])
        pmlxyzppp = self._add_pml(s, [T[0], T[1], T[2]])
        pmlxyzppm = self._add_pml(s, [T[0], T[1], -T[2]])
        pmlxyzpmp = self._add_pml(s, [T[0], -T[1], T[2]])
        pmlxyzpmm = self._add_pml(s, [T[0], -T[1], -T[2]])
        pmlxyzmpp = self._add_pml(s, [-T[0], T[1], T[2]])
        pmlxyzmpm = self._add_pml(s, [-T[0], T[1], -T[2]])
        pmlxyzmmp = self._add_pml(s, [-T[0], -T[1], T[2]])
        pmlxyzmmm = self._add_pml(s, [-T[0], -T[1], -T[2]])

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
        self._translate([self.box] + self.pmls, self.box_center)
        self.fragment(self.box, self.pmls)

        if Rcalc > 0:
            sphere_calc = self.add_sphere(*self.box_center, Rcalc)
            box, sphere_calc = self.fragment(box, sphere_calc)
            self.box = [box, sphere_calc]

        self.add_physical(box, "box")
        self.add_physical(pmlx, "pmlx")
        self.add_physical(pmly, "pmly")
        self.add_physical(pmlz, "pmlz")
        self.add_physical(pmlxy, "pmlxy")
        self.add_physical(pmlyz, "pmlyz")
        self.add_physical(pmlxz, "pmlxz")
        self.add_physical(pml3, "pmlxyz")

        self.pml_physical = [
            "pmlx",
            "pmly",
            "pmlz",
            "pmlxy",
            "pmlyz",
            "pmlxz",
            "pmlxyz",
        ]

        if Rcalc > 0:
            bnds = self.get_boundaries("box")
            self.calc_bnds = bnds[0]
            self.add_physical(self.calc_bnds, "calc_bnds", dim=2)

    def _addbox_center(self, rect_size):
        corner = -np.array(rect_size) / 2
        corner = tuple(corner)
        return self.add_box(*corner, *rect_size)

    def _translate(self, tag, t):
        translation = tuple(t)
        self.translate(self.dimtag(tag), *translation)

    def _add_pml(self, s, t):
        pml = self._addbox_center(s)
        self._translate(pml, t)
        return pml
