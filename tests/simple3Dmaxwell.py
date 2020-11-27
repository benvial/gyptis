#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from math import pi

import dolfin as df

from gyptis.complex import *
from gyptis.geometry import *
from gyptis.materials import *
from gyptis.sources import *

#
#
# model = Model("test")
# box = model.addBox(0, 0, 0, 1, 1, 2)
# other = model.addBox(0, 0, 1, 1, 1, 1)
# # other = model.fragment(model.dimtag(other),model.dimtag(box))
# other,box = model.fragmentize(other,box)
#
# model.add_physical(box, "box")
# model.add_physical(other, "other")
# model.build(True)
#
# cdsdc

model = Model("test")
box = model.addBox(0, 0, 0, 1, 1, 1)
other = model.addBox(0, 0, 0, 1, 1, 2)
box, other = model.fragmentize(box, other)
sphere = model.addSphere(0.5, 0.5, 0.5, 0.1)
sphere, box = model.fragmentize(sphere, box)
model.set_size(box, 0.1)
model.set_size(sphere, 0.1)
model.set_size(other, 0.1)
model.add_physical(sphere, "sphere")
model.add_physical(box, "box")
model.add_physical(other, "other")
mesh_object = model.build()
mesh = model.mesh_object["mesh"]
markers = model.mesh_object["markers"]["tetra"]
domains = model.subdomains["volumes"]
dx = model.measure["dx"]
#
# nmesh = 16
# mesh = df.UnitCubeMesh(nmesh, nmesh, nmesh)

lambda0 = 0.5
k0 = 2 * pi / lambda0
X2 = [f"pow((x[{i}]-0.5)/0.1,2)" for i in range(3)]
Z = "+".join(X2)
s = f"exp(-({Z}))"
S = df.Expression((s, "0", "0"), degree=0, domain=mesh)

S = plane_wave_3D(lambda0, 0, 0, 0, domain=mesh)

e = (3 * np.eye(3, dtype=complex)).tolist()
m = (np.eye(3, dtype=complex)).tolist()

epsilon = {"sphere": e, "box": m, "other": m}
epsilon_doms = Subdomain(markers, domains, epsilon, degree=1)


epsilon_annex = {"sphere": m, "box": m, "other": m}
epsilon_doms_annex = Subdomain(markers, domains, epsilon_annex, degree=1)


#
# W = df.FunctionSpace(mesh, "N1curl", 1)
# dx = df.dx
# E = df.Function(W)
# Etrial = df.TrialFunction(W)
# Etest = df.TestFunction(W)
#
# L = -df.inner(df.curl(Etrial), df.curl(Etest)) * dx + k0**2 * df.inner(Etrial, Etest) * dx
# b = -df.dot(S, Etest) * dx
#
# Ah = df.assemble(L)
# bh = df.assemble(b)
# solver = df.LUSolver(Ah)
# solver.solve(Ah, E.vector(), bh)
#
# W0 = df.FunctionSpace(mesh, "DG", 0)
# df.File("test.pvd") << df.project(E[0], W0)


W = ComplexFunctionSpace(mesh, "N1curl", 1, constrained_domain=None)
# dx = df.dx
E = Function(W)
Etrial = TrialFunction(W)
Etest = TestFunction(W)


deps = epsilon_doms - epsilon_doms_annex

L = (
    -inner(curl(Etrial), curl(Etest)) * dx
    + k0 ** 2 * inner(epsilon_doms * Etrial, Etest) * dx
)
b = -dot(deps * S, Etest) * dx

L = L.real + L.imag
b = b.real + b.imag

Ah = assemble(L)
bh = assemble(b)
solver = df.LUSolver(Ah)


Efunc = E[0].real.ufl_operands[0]
solver.solve(Ah, Efunc.vector(), bh)

W0 = df.FunctionSpace(mesh, "DG", 0)
df.File("test.pvd") << project(E[0].real, W0)
# df.File("test.pvd") << project(E[2].real, W0)
# df.File("test.pvd") << project(E[0].imag, W0)
