#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io
import pyvista

reader = pyvista.get_reader("reEx.pvd")
mesh = reader.read()
pl = pyvista.Plotter()
_ = pl.add_mesh(
    mesh,
    cmap="RdBu_r",
    interpolate_before_map=True,
    scalar_bar_args={"title": "Re E", "vertical": True},
)
pl.view_yz()
pl.show()
