# -*- coding: utf-8 -*-
"""
Sphere
======

An example of scattering from a dielectric sphere
"""

from gyptis import geometry

##############################################################################
# We first define the geometry

model = geometry.Model("Scattering from a sphere")
box = model.addBox(-1, -1, -1, 2, 2, 2)
sphere = model.addSphere(0, 0, 0, 0.5)
sphere, box = model.fragmentize(sphere, box)
model.set_size(box, 0.1)
model.set_size(sphere, 0.1)
outer_bnds = model.get_boundaries(box)

model.add_physical(sphere, "sphere")
model.add_physical(box, "box")
model.add_physical(outer_bnds, "outer_bnds", dim=2)

mesh_info = model.build(interactive=False)
