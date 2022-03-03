#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

from .geometry import *


class Lattice3D(Geometry):
    def __init__(
        self,
        vectors,
        periodic_tol=1e-8,
        **kwargs,
    ):
        super().__init__(
            dim=3,
            **kwargs,
        )

        self.periodic_tol = periodic_tol
        self.vectors = vectors
        v = self.vectors
        self.vertices = [
            (0, 0, 0),
            (v[0][0], v[0][1], v[0][2]),
            (
                v[0][0] + v[1][0],
                v[0][1] + v[1][1],
                v[0][2] + v[1][2],
            ),
            (v[1][0], v[1][1], v[1][2]),
            (v[2][0], v[2][1], v[2][2]),
            (v[2][0] + v[0][0], v[2][1] + v[0][1], v[2][2] + v[0][2]),
            (
                v[2][0] + v[0][0] + v[1][0],
                v[2][1] + v[0][1] + v[1][1],
                v[2][2] + v[0][2] + v[1][2],
            ),
            (v[2][0] + v[1][0], v[2][1] + v[1][1], v[2][2] + v[1][2]),
        ]
        p = []
        for v in self.vertices:
            p.append(self.add_point(*v))
        l0 = []
        for i in range(3):
            l0.append(self.add_line(p[i + 1], p[i]))
        l0.append(self.add_line(p[3], p[0]))
        cl = self.add_curve_loop(l0)
        ps0 = self.add_plane_surface([cl])

        l1 = []
        for i in range(4, 7):
            l1.append(self.add_line(p[i + 1], p[i]))
        l1.append(self.add_line(p[7], p[4]))
        cl = self.add_curve_loop(l1)
        ps1 = self.add_plane_surface([cl])
        #
        #
        l2 = []
        l2.append(l0[0])
        l2.append(self.add_line(p[1], p[5]))
        l2.append(l1[0])
        l2.append(self.add_line(p[4], p[0]))
        cl = self.add_curve_loop(l2)
        ps2 = self.add_plane_surface([cl])

        l3 = []
        l3.append(l0[2])
        l3.append(self.add_line(p[3], p[7]))
        l3.append(l1[2])
        l3.append(self.add_line(p[6], p[2]))
        cl = self.add_curve_loop(l3)
        ps3 = self.add_plane_surface([cl])

        l4 = []
        l4.append(l2[3])
        l4.append(l0[3])
        l4.append(l3[1])
        l4.append(l1[3])
        cl = self.add_curve_loop(l4)
        ps4 = self.add_plane_surface([cl])

        l5 = []
        l5.append(l2[1])
        l5.append(l0[1])
        l5.append(l3[3])
        l5.append(l1[1])
        cl = self.add_curve_loop(l5)
        ps5 = self.add_plane_surface([cl])
        self.perbnds = [ps0, ps1, ps2, ps3, ps4, ps5]
        sl = self.add_surface_loop(self.perbnds)
        self.cell = self.add_volume([sl])
        self.add_physical(self.cell, "cell")

        # self.cell = ps
        # self.add_physical(self.cell, "cell")

    @property
    def translation(self):
        return (
            self._translation_matrix([*self.vectors[0]]),
            self._translation_matrix([*self.vectors[1]]),
            self._translation_matrix([*self.vectors[2]]),
        )

    def get_periodic_bnds(self):
        verts = self.vertices
        # get all boundaries
        bnds = self.get_entities(2)
        maps = []
        p0 = verts[0], verts[1], verts[2]
        p1 = verts[4], verts[5], verts[6]
        p2 = verts[0], verts[1], verts[4]
        p3 = verts[2], verts[3], verts[6]
        p4 = verts[0], verts[3], verts[4]
        p5 = verts[1], verts[2], verts[6]
        self.planes = [p0, p1, p2, p3, p4, p5]
        for pl in self.planes:
            wheres = []
            for b in bnds:
                qb = gmsh.model.getParametrizationBounds(2, b[-1])
                B = []
                for p in qb:
                    P = gmsh.model.getValue(2, b[-1], p)
                    belongs = is_on_plane(P, *pl, eps=self.periodic_tol)
                    B.append(belongs)
                alls = np.all(B)
                if alls:
                    wheres.append(b)
            maps.append(wheres)
        s = {}
        s["-0"] = [m[-1] for m in maps[4]]
        s["+0"] = [m[-1] for m in maps[5]]
        s["-1"] = [m[-1] for m in maps[2]]
        s["+1"] = [m[-1] for m in maps[3]]
        s["-2"] = [m[-1] for m in maps[0]]
        s["+2"] = [m[-1] for m in maps[1]]
        return s

    def build(self, *args, periodic=True, **kwargs):
        if periodic:
            periodic_id = self.get_periodic_bnds()
            gmsh.model.mesh.setPeriodic(
                2, periodic_id["+0"], periodic_id["-0"], self.translation[0]
            )
            gmsh.model.mesh.setPeriodic(
                2, periodic_id["+1"], periodic_id["-1"], self.translation[1]
            )
            gmsh.model.mesh.setPeriodic(
                2, periodic_id["+2"], periodic_id["-2"], self.translation[2]
            )
        super().build(*args, **kwargs)


class CubicLattice(Lattice3D):
    def __init__(
        self,
        lattice_constant,
        **kwargs,
    ):
        self.lattice_constant = lattice_constant
        vectors = (lattice_constant, lattice_constant, lattice_constant)
        super().__init__(
            vectors,
            **kwargs,
        )
