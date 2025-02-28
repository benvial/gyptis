#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

import gmsh

gmsh.initialize()


fact = gmsh.model.occ
r1 = fact.add_rectangle(0, 0, 0, 1, 1)
r2 = fact.add_rectangle(0, 1, 0, 1, 1)
fact.remove_all_duplicates()
r3 = fact.add_rectangle(0.25, 0.25, 0, 0.5, 0.5)
r4, r5 = fact.fragment([(2, r1)], [(2, r3)])
fact.synchronize()

# fact.remove_all_duplicates()
# fact.synchronize()
# gmsh.model.mesh.setPeriodic(1, [7,9], [5,10],
#                             [1.0, 0.0, 0.0, 8.0, 0.0, 1.0, 0.0, 0, 0.0, 0.0, 1.0, 0, 0.0, 0.0, 0.0, 1.0])

gmsh.model.mesh.generate(2)
gmsh.write("test.msh")
gmsh.fltk.run()
gmsh.finalize()
