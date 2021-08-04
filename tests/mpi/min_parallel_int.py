#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import dolfin
from dolfin import *

dolfin.parameters["ghost_mode"] = "shared_vertex"
dolfin.parameters["ghost_mode"] = "shared_facet"
N = 54
# mesh = UnitCubeMesh(MPI.comm_self, N, N, N)
mesh = UnitCubeMesh(N, N, N)

facet_f = MeshValueCollection("size_t", mesh, 2)
mark = dolfin.cpp.mesh.MeshFunctionSizet(mesh, facet_f)
CompiledSubDomain("near(x[0], 0.5)").mark(mark, 1)

ex = Constant((1, 0, 0))

n = FacetNormal(mesh)
s = 1  # -dot(n("+"), n("+"))

form = s * n[0]("+") * dS(domain=mesh, subdomain_data=mark, subdomain_id=1)
# form = s*0.5*abs(n('+')[0] - n('-')[0])* dS(domain=mesh, subdomain_data=mark, subdomain_id=1)
# form = jump(n[0])* dS(domain=mesh, subdomain_data=mark, subdomain_id=1)
# form = 1 * dS(domain=mesh, subdomain_data=mark, subdomain_id=1)
# form = dot(ex,n)('+') * dS(domain=mesh, subdomain_data=mark, subdomain_id=1)
# form = 0.5*(n('+')[0] + n('-')[0])* dS(domain=mesh, subdomain_data=mark, subdomain_id=1)
value = assemble(form)
print(value)
