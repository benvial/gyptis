# from dolfin import *
# from dolfin_adjoint import *


from dolfin import *

import gyptis

# gyptis.adjoint(True)
from gyptis import dolfin
from gyptis.physics import *

DirichletBC = dolfin.DirichletBC

k0_ = 10
k0 = Constant(k0_)

n = 30
mesh = UnitSquareMesh(n, n)


V = ComplexFunctionSpace(mesh, "CG", 2)
Vre = FunctionSpace(mesh, "CG", 2)
# Vre = V.sub(0).collapse()
# Vim = V.sub(1).collapse()
VV = VectorFunctionSpace(mesh, "CG", 2)

ctrl = dolfin.Expression("sin(3*2*pi*x[1]*x[0])", degree=2)
ctrl = dolfin.project(ctrl, Vre)

eps = ctrl * Complex(2, 7) + 4

# eps = project(eps, Vre)

s = plane_wave_2D(k0_, 0.4, domain=mesh, grad=False)

u = TrialFunction(V)
v = TestFunction(V)

bc = DirichletBC(V, (0, 0), "on_boundary")

A_ = [-inner(grad(u), grad(v)) * dx, k0 ** 2 * eps * inner((u), (v)) * dx]
b_ = k0 ** 2 * (eps * s * v) * dx
A_ = [a_.real + a_.imag for a_ in A_]
b_ = b_.real + b_.imag
Ah = [assemble(a_) for a_ in A_]
bh = assemble(b_)
Ah_ = Ah[0] + Ah[1]
Ah_.form = Ah[0].form + Ah[1].form

# bc.apply(Ah_, bh)

u = dolfin.Function(VV)
uvec = u


# u = Function(V)
# uvec = u.real


solver = LUSolver("mumps")
dx = Measure("dx", mesh)

# J = assemble(inner(u, u.conj) * dx).real

solver.solve(Ah_, u.vector(), bh)

u = Complex(*u.split())
J = assemble(inner(u, u.conj) * dx).real
# J = assemble(inner(u, u) * dx)#.real
dJdu = compute_gradient(J, Control(ctrl))

import matplotlib.pyplot as plt

# u = project(u, Vre)

plt.clf()
plot(u.real)
plt.show()
plt.figure()
plot(dJdu)
plt.show()
