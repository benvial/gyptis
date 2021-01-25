#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from math import pi

import dolfin

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
# mesh = dolfin.UnitCubeMesh(nmesh, nmesh, nmesh)

lambda0 = 0.5
k0 = 2 * pi / lambda0
X2 = [f"pow((x[{i}]-0.5)/0.1,2)" for i in range(3)]
Z = "+".join(X2)
s = f"exp(-({Z}))"
S = dolfin.Expression((s, "0", "0"), degree=0, domain=mesh)

S = plane_wave_3D(lambda0, 0, 0, 0, domain=mesh)

e = (3 * np.eye(3, dtype=complex)).tolist()
m = (np.eye(3, dtype=complex)).tolist()

epsilon = {"sphere": e, "box": m, "other": m}
epsilon_doms = Subdomain(markers, domains, epsilon, degree=1)


epsilon_annex = {"sphere": m, "box": m, "other": m}
epsilon_doms_annex = Subdomain(markers, domains, epsilon_annex, degree=1)


#
# W = dolfin.FunctionSpace(mesh, "N1curl", 1)
# dx = dolfin.dx
# E = dolfin.Function(W)
# Etrial = dolfin.TrialFunction(W)
# Etest = dolfin.TestFunction(W)
#
# L = -dolfin.inner(dolfin.curl(Etrial), dolfin.curl(Etest)) * dx + k0**2 * dolfin.inner(Etrial, Etest) * dx
# b = -dolfin.dot(S, Etest) * dx
#
# Ah = dolfin.assemble(L)
# bh = dolfin.assemble(b)
# solver = dolfin.LUSolver(Ah)
# solver.solve(Ah, E.vector(), bh)
#
# W0 = dolfin.FunctionSpace(mesh, "DG", 0)
# dolfin.File("test.pvd") << dolfin.project(E[0], W0)


W = ComplexFunctionSpace(mesh, "N1curl", 1, constrained_domain=None)
# dx = dolfin.dx
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
solver = dolfin.LUSolver(Ah)


Efunc = E[0].real.ufl_operands[0]
solver.solve(Ah, Efunc.vector(), bh)

W0 = dolfin.FunctionSpace(mesh, "DG", 0)
dolfin.File("test.pvd") << project(E[0].real, W0)
# dolfin.File("test.pvd") << project(E[2].real, W0)
# dolfin.File("test.pvd") << project(E[0].imag, W0)
