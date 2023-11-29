#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


from . import dolfin
from .geometry import *


def prepare_boundary_conditions(bc_dict):
    valid_bcs = ["PEC"]
    pec_bnds = []
    for bnd, cond in bc_dict.items():
        if cond not in valid_bcs:
            raise ValueError(f"Unknown boundary condition {cond}")
        else:
            pec_bnds.append(bnd)
    return pec_bnds


def build_pec_boundary_conditions(pec_bnds, geometry, function_space, applied_function):
    boundary_conditions = []
    for bnd in pec_bnds:
        bc = DirichletBC(
            function_space,
            applied_function,
            geometry.boundary_markers,
            bnd,
            geometry.boundaries,
        )
        [boundary_conditions.append(b) for b in bc]
    return boundary_conditions


class _DirichletBC(dolfin.DirichletBC):
    def __init__(self, *args, **kwargs):
        self.subdomain_dict = args[-1]
        if not callable(args[2]):
            args = list(args)
            args[-2] = self.subdomain_dict[args[-2]]
            args = tuple(args[:-1])
        super().__init__(*args)


class DirichletBC:
    def __new__(cls, *args, **kwargs):
        W = args[0]
        value = args[1]
        Wre, Wim = W.split()
        bcre = _DirichletBC(Wre, value.real, *args[2:], **kwargs)
        bcim = _DirichletBC(Wim, value.imag, *args[2:], **kwargs)
        return bcre, bcim


class BiPeriodic2D(dolfin.SubDomain):
    def __init__(self, geometry, eps=dolfin.DOLFIN_EPS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.vectors = geometry.vectors
        self.vertices = geometry.vertices

    def inside(self, x, on_boundary):
        on_bottom = is_on_line(x, self.vertices[0], self.vertices[1], eps=self.eps)
        on_left = is_on_line(x, self.vertices[3], self.vertices[0], eps=self.eps)

        on_vert_1 = dolfin.near(
            x[0], self.vertices[1][0], eps=self.eps
        ) and dolfin.near(x[1], self.vertices[1][1], eps=self.eps)
        on_vert_3 = dolfin.near(
            x[0], self.vertices[3][0], eps=self.eps
        ) and dolfin.near(x[1], self.vertices[3][1], eps=self.eps)
        return bool(
            (on_bottom or on_left) and not on_vert_1 and not on_vert_3 and on_boundary
        )

    def map(self, x, y):
        on_right = is_on_line(x, self.vertices[1], self.vertices[2], eps=self.eps)
        on_top = is_on_line(x, self.vertices[3], self.vertices[2], eps=self.eps)

        if on_right and on_top:
            y[0] = x[0] - self.vectors[0][0] - self.vectors[1][0]
            y[1] = x[1] - self.vectors[0][1] - self.vectors[1][1]
        elif on_right:
            y[0] = x[0] - self.vectors[0][0]
            y[1] = x[1] - self.vectors[0][1]
        elif on_top:
            y[0] = x[0] - self.vectors[1][0]
            y[1] = x[1] - self.vectors[1][1]
        else:
            y[0] = -10000
            y[1] = -10000


class PeriodicBoundary2DX(dolfin.SubDomain):
    def __init__(self, period, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period

    def inside(self, x, on_boundary):
        return bool(dolfin.near(x[0], -self.period / 2) and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - self.period
        y[1] = x[1]


class BiPeriodicBoundary3D(dolfin.SubDomain):
    def __init__(self, period, eps=dolfin.DOLFIN_EPS, map_tol=1e-10):
        self.period = period
        self.eps = eps
        self.map_tol = map_tol
        super().__init__(map_tol=map_tol)

    def near(self, x, y):
        return dolfin.near(x, y, eps=self.eps)

    def on_it(self, x, i, s="-"):
        s = -1 if s == "-" else 1
        return self.near(x[i], s * self.period[i] / 2)

    def inside(self, x, on_boundary):
        if on_boundary:
            if self.on_it(x, 0, "-") and not self.on_it(x, 1, "+"):
                return True
            return bool(self.on_it(x, 1, "-") and not self.on_it(x, 0, "+"))

    def map(self, x, y):
        y[0] = x[0] - self.period[0] if self.on_it(x, 0, "+") else x[0]
        y[1] = x[1] - self.period[1] if self.on_it(x, 1, "+") else x[1]
        y[2] = x[2]


class Periodic3D(dolfin.SubDomain):
    def __init__(self, geometry, eps=dolfin.DOLFIN_EPS, map_tol=1e-10):
        self.eps = eps
        self.map_tol = map_tol
        self.vectors = geometry.vectors
        self.vertices = geometry.vertices
        self.planes = geometry.planes
        super().__init__(map_tol=map_tol)

    def inside(self, x, on_boundary):
        ison0 = is_on_plane(x, *self.planes[4], eps=self.eps)
        ison1 = is_on_plane(x, *self.planes[2], eps=self.eps)
        ison2 = is_on_plane(x, *self.planes[0], eps=self.eps)

        ison_slave0 = is_on_line3D(x, self.planes[4], self.planes[2], eps=self.eps)
        ison_slave1 = is_on_line3D(x, self.planes[4], self.planes[0], eps=self.eps)
        ison_slave2 = is_on_line3D(x, self.planes[2], self.planes[0], eps=self.eps)

        return bool(
            (ison0 or ison1 or ison2)
            and (not ((ison_slave0) or (ison_slave1) or (ison_slave2)))
            and on_boundary
        )

    def map(self, x, y):
        v = self.vectors

        def set_coord(x, y):
            for i in range(3):
                y[i] = x[i]
            return y

        ison0 = is_on_plane(x, *self.planes[5], self.map_tol)
        ison1 = is_on_plane(x, *self.planes[3], self.map_tol)
        ison2 = is_on_plane(x, *self.planes[1], self.map_tol)
        if ison0 and ison1 and ison2:
            y = set_coord(x - v[0] - v[1] - v[2], y)

        elif ison0 and ison2:
            y = set_coord(x - v[0] - v[2], y)
        elif ison1 and ison2:
            y = set_coord(x - v[1] - v[2], y)
        elif ison0 and ison1:
            y = set_coord(x - v[0] - v[1], y)
        elif ison0:
            y = set_coord(x - v[0], y)
        elif ison1:
            y = set_coord(x - v[1], y)
        elif ison2:
            y = set_coord(x - v[2], y)
        else:
            y = set_coord([10000, 10000, 10000], y)
