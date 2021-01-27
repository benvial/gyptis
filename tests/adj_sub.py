# from dolfin import *
# from dolfin_adjoint import *


import matplotlib.pyplot as plt
from test_geometry import geom2D

from gyptis.complex import *
from gyptis.geometry import *
from gyptis.materials import *
from gyptis.physics import *

pi = np.pi

parmesh = 10
lambda0 = 1
eps_cyl0 = 3
eps_box = 1
k0 = 2 * np.pi / lambda0
# k0 = Constant(k0_)


geom = BoxPML(dim=2, box_size=(3, 3), pml_width=(1, 1))
r = 1
cyl = geom.add_disk(0, 0, 0, r, r)
cyl, geom.box = geom.fragment(cyl, geom.box)
geom.add_physical(geom.box, "box")
geom.add_physical(cyl, "cyl")
pmls = [d for d in geom.subdomains["surfaces"] if d.startswith("pml")]
geom.set_size(pmls, lambda0 / parmesh * 0.7)
geom.set_size("box", lambda0 / (parmesh))
geom.set_size("cyl", lambda0 / (parmesh * eps_cyl0 ** 0.5))

geom.build(interactive=False)
model = geom

mesh = model.mesh_object["mesh"]
dx = model.measure["dx"]
markers = model.mesh_object["markers"]["triangle"]
domains = model.subdomains["surfaces"]
# sub_py = Subdomain(markers, domains, values, degree=1, cpp=False)

submesh = dolfin.SubMesh(mesh, markers, domains["cyl"])
submesh = mesh


V = ComplexFunctionSpace(mesh, "CG", 2)
Vre = dolfin.FunctionSpace(mesh, "CG", 2)
VV = dolfin.VectorFunctionSpace(mesh, "CG", 2)
ASub = dolfin.FunctionSpace(submesh, "DG", 0)
# ASub = Vre

ctrl0 = dolfin.Expression("10", degree=2)
ctrl = project(ctrl0, ASub)

###  works
eps_cyl_func = ctrl  # * Complex(2.5, 0.5) + 6

## doesnt work
# eps_cyl_func = ctrl0 * Complex(2.5, 0.5) + 6
# eps_cyl_func = project(eps_cyl_func,Vre)

# eps = dict(cyl=eps_cyl_func, box=2)

eps = {d: 1 for d in domains}
eps["cyl"] = eps_cyl_func  # * Complex(1, 0.01)


u0 = plane_wave_2D(k0, 0.0, domain=mesh, grad=False)

u = TrialFunction(V)
v = TestFunction(V)

# bc = DirichletBC(V, Complex(0, 0), "on_boundary")

# A_ = [-inner(grad(u), grad(v)) * dx("box")]
# A_.append(-inner(grad(u), grad(v)) * dx("cyl"))
#
# A_.append(k0 ** 2 * eps["box"] * inner((u), (v)) * dx("box"))
# A_.append(k0 ** 2 * eps["cyl"] * inner((u), (v)) * dx("cyl"))

A_ = [
    (-inner(grad(u), grad(v)) + k0 ** 2 * eps[d] * inner((u), (v))) * dx(d)
    for d in domains
]

b_ = k0 ** 2 * (u0 * (eps["cyl"] - eps["box"]) * v) * dx("cyl")
A_ = [a_.real + a_.imag for a_ in A_]
b = b_.real + b_.imag
A = [assemble(a_) for a_ in A_]
bh = assemble(b)
Ah = sum(A[1:], start=A[0])

Ah.form = A[0].form
for a in A[1:]:
    Ah.form += a.form


# bc.apply(Ah_, bh)

u = dolfin.Function(VV)
uvec = u

# u = Function(V)
# uvec = u.real

solver = dolfin.LUSolver("mumps")

# J = assemble(inner(u, u.conj) * dx).real

solver.solve(Ah, uvec.vector(), bh)

u = Complex(*u.split())
J = assemble(inner(u, u.conj) * dx("box")).real
# J = assemble(inner(u, u) * dx)#.real
dJdu = dolfin.compute_gradient(J, dolfin.Control(ctrl))


# u = project(u, Vre)

plt.clf()
cb = dolfin.plot(u.real)
plt.colorbar(cb)
plt.show()
plt.figure()
cb = dolfin.plot(dJdu)
plt.colorbar(cb)
plt.show()
