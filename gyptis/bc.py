#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from . import dolfin


class _DirichletBC(dolfin.DirichletBC):
    def __init__(self, *args, **kwargs):
        self.subdomain_dict = args[-1]
        if not callable(args[2]):
            args = list(args)
            args[-2] = self.subdomain_dict[args[-2]]
            args = tuple(args[:-1])
        super().__init__(*args)


class DirichletBC:
    def __new__(self, *args, **kwargs):
        W = args[0]
        value = args[1]
        Wre, Wim = W.split()
        bcre = _DirichletBC(Wre, value.real, *args[2:], **kwargs)
        bcim = _DirichletBC(Wim, value.imag, *args[2:], **kwargs)
        return bcre, bcim


def _on_bnd(p, p1, p2):
    x, y = p
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return dolfin.near(x, x1)
    else:
        return dolfin.near(y - y1, (y2 - y1) / (x2 - x1) * (x - x1))


class BiPeriodic2D(dolfin.SubDomain):
    def __init__(self, vectors, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectors = vectors
        self.vertices = [
            (0, 0),
            (self.vectors[0][0], self.vectors[0][1]),
            (
                self.vectors[0][0] + self.vectors[1][0],
                self.vectors[0][1] + self.vectors[1][1],
            ),
            (self.vectors[1][0], self.vectors[1][1]),
        ]

    def inside(self, x, on_boundary):
        on_bottom = _on_bnd(x, self.vertices[0], self.vertices[1])
        on_left = _on_bnd(x, self.vertices[3], self.vertices[0])

        on_vert_0 = dolfin.near(x[0], self.vertices[0][0]) and dolfin.near(
            x[1], self.vertices[0][1]
        )
        on_vert_3 = dolfin.near(x[0], self.vertices[3][0]) and dolfin.near(
            x[1], self.vertices[3][1]
        )

        return bool(
            (on_bottom or on_left) and (not (on_vert_0 or on_vert_3)) and on_boundary
        )

    def map(self, x, y):
        verts = self.vertices.copy()
        verts.append(self.vertices[0])
        on_right = _on_bnd(x, self.vertices[1], self.vertices[2])
        on_top = _on_bnd(x, self.vertices[2], self.vertices[3])
        # if on_right and on_top:
        #     y[0] = x[0] - self.vectors[0][0]
        #     y[1] = x[1] - self.vectors[1][1]
        if on_right:
            y[0] = x[0] - self.vectors[0][0]
            y[1] = x[1] - self.vectors[0][1]
        elif on_top:
            y[0] = x[0] - self.vectors[1][0]
            y[1] = x[1] - self.vectors[1][1]
        else:
            y[0] = -1000
            y[1] = -1000


class PeriodicBoundary2DX(dolfin.SubDomain):
    def __init__(self, period, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period

    def inside(self, x, on_boundary):
        return bool(dolfin.near(x[0], -self.period / 2) and on_boundary)

    # # Left boundary is "target domain" G
    # def inside(self, x, on_boundary):
    #     return bool(
    #         x[0] - self.period / 2 < dolfin.DOLFIN_EPS
    #         and x[0] - self.period / 2 > -dolfin.DOLFIN_EPS
    #         and on_boundary
    #     )

    def map(self, x, y):
        y[0] = x[0] - self.period
        y[1] = x[1]

    #
    # def map(self, x, y):
    #     if dolfin.near(x[0], self.period / 2):
    #         y[0] = x[0] - self.period
    #         y[1] = x[1]
    #     else:
    #         y[0] = -1000
    #         y[1] = -1000


class BiPeriodicBoundary3D(dolfin.SubDomain):
    def __init__(self, period, **kwargs):
        self.period = period
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        return bool(
            (
                dolfin.near(x[0], -self.period[0] / 2)
                or dolfin.near(x[1], -self.period[1] / 2)
            )
            and (
                not (
                    (
                        dolfin.near(x[0], -self.period[0] / 2)
                        and dolfin.near(x[1], self.period[1] / 2)
                    )
                    or (
                        dolfin.near(x[0], self.period[0] / 2)
                        and dolfin.near(x[1], -self.period[1] / 2)
                    )
                )
            )
            and on_boundary
        )

    def map(self, x, y):
        if dolfin.near(x[0], self.period[0] / 2) and dolfin.near(
            x[1], self.period[1] / 2
        ):
            y[0] = x[0] - self.period[0]
            y[1] = x[1] - self.period[1]
            y[2] = x[2]
        elif dolfin.near(x[0], self.period[0] / 2):
            y[0] = x[0] - self.period[0]
            y[1] = x[1]
            y[2] = x[2]
        elif dolfin.near(x[1], self.period[1] / 2):
            y[0] = x[0]
            y[1] = x[1] - self.period[1]
            y[2] = x[2]
        else:
            y[0] = -1000  # -self.period[0]*2.
            y[1] = -1000  # -self.period[1]*2.
            y[2] = -1000  # 0.


def prepare_boundary_conditions(bc_dict):
    valid_bcs = ["PEC"]
    boundary_conditions = []
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
