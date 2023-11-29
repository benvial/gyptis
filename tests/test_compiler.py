#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


def test_flags():
    import os
    import time

    import numpy as np

    import gyptis
    from gyptis import BoxPML, PlaneWave, Scattering
    from gyptis.utils import list_time

    gyptis.dolfin.set_log_level(6)

    # test = "-O3 -fno-align-functions -fno-align-jumps -fno-align-loops -fno-align-labels -march=native"
    # test = "-O3 -march=native"

    flags = ["-O0", "-O1", "-O2", "-O3", "-Ofast", "-Ofast -march=native"]
    flags = ["-O3"]
    tsolve = []
    tint = []

    polarization = "TM"
    degree = 2
    wavelength = 0.3
    pmesh = 3
    for cppflag in flags:
        print(f"-------   { cppflag }   --------")

        gyptis.dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = cppflag

        lmin = wavelength / pmesh

        geom = BoxPML(
            dim=2,
            box_size=(4 * wavelength, 4 * wavelength),
            pml_width=(wavelength, wavelength),
        )
        cyl = geom.add_circle(0, 0, 0, 0.2)
        cyl, box = geom.fragment(cyl, geom.box)
        geom.add_physical(box, "box")
        geom.add_physical(cyl, "cyl")
        [geom.set_size(pml, lmin) for pml in geom.pmls]
        geom.set_size("box", lmin)
        geom.set_size("cyl", lmin)
        geom.build()

        epsilon = dict(box=1, cyl=3)
        mu = dict(box=1, cyl=1)
        pw = PlaneWave(
            wavelength=wavelength, angle=0, dim=2, domain=geom.mesh, degree=degree
        )

        for i in range(1, 2):
            s = Scattering(
                geom, epsilon, mu, pw, degree=degree, polarization=polarization
            )
            print(s.function_space.dim())
            t = -time.time()
            u = s.solve()
            t += time.time()
            print(f"elapsed time solve {t:0.3f}s")
            list_time()
            t1 = -time.time()
            gyptis.assemble(u * s.formulation.dx)
            t1 += time.time()
            print(f"elapsed time integral {t1:0.3f}s")
            # list_time()

            if i == 1:
                # skip the first as JIT compilation may happen
                tsolve.append(t)
                tint.append(t1)

    for cppflag, t, t1 in zip(flags, tsolve, tint):
        print(f"-------   { cppflag }   --------")
        print(f"elapsed time solve {t:0.3f}s")
        print(f"elapsed time integral {t1:0.3f}s")
