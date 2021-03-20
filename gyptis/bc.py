#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from . import dolfin


class DirichletBC(dolfin.DirichletBC):
    def __init__(self, *args, **kwargs):
        self.subdomain_dict = args[-1]
        if not callable(args[2]):
            args = list(args)
            args[-2] = self.subdomain_dict[args[-2]]
            args = tuple(args[:-1])
        super().__init__(*args)


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
