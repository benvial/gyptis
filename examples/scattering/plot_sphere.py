# -*- coding: utf-8 -*-
"""
Sphere
======

An example of scattering from a dielectric sphere
"""

from gyptis.geometry import BoxPML3D

##############################################################################
# We first define the geometry

model = BoxPML3D(name="Scattering from a sphere")


# box = model.addBox(-1, -1, -1, 2, 2, 2)
sphere = model.addSphere(0, 0, 0, 0.3)
sphere, box = model.fragmentize(sphere, model.box)
model.set_size(box, 0.1)
model.set_size(sphere, 0.1)

model.add_physical(sphere, "sphere")
model.add_physical(box, "box")

# model.set_size("box",0.1)
model.set_size("sphere", 0.05)
# model.set_size("sphere",0.2)

# outer_bnds = [model.get_boundaries(p,physical=False) for p in model.pmls]

# outer_bnds = [a for b in outer_bnds for a in b]

# model.add_physical(outer_bnds, "outer_bnds", dim=2)

mesh_info = model.build()
