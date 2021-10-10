#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import time

from gyptis import BoxPML, Scattering
from gyptis.source import PlaneWave
from gyptis.utils.helpers import list_time


def test_simple(pmesh=10):
    polarization = "TM"
    degree = 2

    wavelength = 0.3
    options = {"General.NumThreads": 0, "Mesh.Algorithm": 6, "General.Verbosity": 2}
    geom = BoxPML(
        dim=2,
        box_size=(4 * wavelength, 4 * wavelength),
        pml_width=(wavelength, wavelength),
        options=options,
    )
    cyl = geom.add_circle(0, 0, 0, 0.2)
    cyl, box = geom.fragment(cyl, geom.box)
    geom.add_physical(box, "box")
    geom.add_physical(cyl, "cyl")
    geom.set_size("box", wavelength / pmesh)
    geom.set_size("cyl", wavelength / pmesh)
    t = -time.time()
    geom.build()
    t += time.time()
    print(f"mesh time: {t:.3f}s")
    epsilon = dict(box=1, cyl=3)
    mu = dict(box=1, cyl=1)
    pw = PlaneWave(
        wavelength=wavelength, angle=0, dim=2, domain=geom.mesh, degree=degree
    )
    s = Scattering(geom, epsilon, mu, pw, degree=degree, polarization=polarization)
    u = s.solve()
    list_time()


if __name__ == "__main__":
    import sys

    test_simple(pmesh=float(sys.argv[1]))
