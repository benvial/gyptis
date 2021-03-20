#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from gyptis.materials import *
from gyptis.materials import _get_chi, _get_xi

# # from gyptis import complex as gc
#
# from gyptis import geometry
#
# import importlib
# importlib.reload(geometry)
# from gyptis.geometry import *
#
#
#
# from gyptis import formulation
# importlib.reload(formulation)
# from gyptis.formulation import *


# geom.build(1,1,0,0,0)
#
#
#
# # mesh = geom.mesh_object["mesh"]
#
# domains = dict(all=["box", "cyl"], boundaries=["cyl_bnds"], sources=["cyl"])
#
# W = gc.ComplexFunctionSpace(mesh, "CG", 2)
#
#
# form = Formulation("Test", W, domains)
# coefficients = {}
#
#
# class Coeffs:
#     def __init__(epsilon, mu):
#         self.epsilon = epsilon
#         self.mu = mu
#         self.xi = make_constant_property(xi, dim=2)
#         self.chi = make_constant_property(chi, dim=2)
#
#
# xi = dict(box=1, cyl=3)
# xi = make_constant_property(xi, dim=2)
# coefficients["xi"] = xi
#
# chi = dict(box=10, cyl=0.5)
# chi = make_constant_property(chi, dim=2)
# coefficients["chi"] = chi
#
# parameters = dict(wavenumber=19)
#
# maxwell = Maxwell2D(W, domains, coefficients, parameters)
# maxwell.build_lhs()
#
#
# b = WeakFormBuilder(maxwell, geometry.measure)
#
# b.build_lhs()
