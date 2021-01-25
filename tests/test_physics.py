#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import numpy as np
import pytest

import gyptis
from gyptis import dolfin
from gyptis.physics import *
from gyptis.plotting import *

# dolfin.set_log_level(20)


def test_scatt2D():
    # if __name__ == "__main__":

    pi = np.pi

    parmesh = 4
    lambda0 = 1
    eps_cyl0 = 3
    eps_box = 1

    # eps_cyl = dolfin.Constant(3)

    geom = BoxPML(dim=2, box_size=(6, 6), pml_width=(1, 1))
    r = 0.9
    cyl = geom.addDisk(0, 0, 0, r, r)
    cyl, geom.box = geom.fragmentize(cyl, geom.box)
    geom.add_physical(geom.box, "box")
    geom.add_physical(cyl, "cyl")

    pmls = [d for d in geom.subdomains["surfaces"] if d.startswith("pml")]
    geom.set_size(pmls, lambda0 / parmesh * 0.7)
    geom.set_size("box", lambda0 / (parmesh))
    geom.set_size("cyl", lambda0 / (parmesh * eps_cyl0 ** 0.5))

    geom.build(interactive=False)

    epsilon = dict(cyl=eps_cyl0, box=eps_box)
    mu = dict(cyl=1, box=1)

    def ellipse(Rinclx, Rincly, rot_incl, x0, y0):
        c, s = np.cos(rot_incl), np.sin(rot_incl)
        Rot = np.array([[c, -s], [s, c]])
        nt = 360
        theta = np.linspace(-pi, pi, nt)
        x = Rinclx * np.sin(theta)
        y = Rincly * np.cos(theta)
        x, y = np.linalg.linalg.dot(Rot, np.array([x, y]))
        points = x + x0, y + y0
        return points

    s = Scatt2D(geom, epsilon, mu, polarization="TE", degree=2)

    for s.polarization in ["TE", "TM"]:
        s.prepare()
        s.weak_form()
        s.assemble()
        s.build_system()
        s.solve()
