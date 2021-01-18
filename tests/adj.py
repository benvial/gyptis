# from dolfin import *
# from dolfin_adjoint import *


from dolfin import *

from gyptis import dolfin
from gyptis.physics import *

n = 130
mesh = UnitSquareMesh(n, n)
V = ComplexFunctionSpace(mesh, "CG", 2)
Vre = FunctionSpace(mesh, "CG", 2)
# Vre = V.sub(0).collapse()
# Vim = V.sub(1).collapse()

# VV = VectorFunctionSpace(mesh, "CG", 2)
vv = df.Function(V)

ctrl_0 = df.Expression("sin(3*2*pi*x[1]*x[0])", degree=2)


ctrl = df.project(ctrl_0, Vre)

eps = ctrl * Complex(2, 7) + 4

# control = Control(ctrl)
#
u = TrialFunction(V)
v = TestFunction(V)

nu = Constant(10)


# bc = DirichletBC(V, Complex(0, 0), "on_boundary")

A_ = [inner(grad(u), grad(v)) * dx, nu * eps * inner((u), (v)) * dx]

b_ = (eps * v) * dx

A_ = [a_.real + a_.imag for a_ in A_]
b_ = b_.real + b_.imag

Ah = [assemble(a_) for a_ in A_]
bh = assemble(b_)


Ah_ = Ah[0] + Ah[1]

Ah_.form = Ah[0].form + Ah[1].form

# bc.apply(Ah_, bh)

ucplx = Function(V)
u = df.Function(V)
# ure = df.Function(Vre)
#
# u = df.Function(Vre), df.Function(Vre)
# u = Complex(u[0],u[1])


ufunc = ucplx.real

solver = LUSolver("mumps")
solver.solve(Ah_, ufunc.vector(), bh)
df.list_timings(df.TimingClear.clear, [df.TimingType.wall])

#
# a = sum(A_)
# u = Function(V)
# solve(a==b_,u, bc)

J = assemble(inner(u, u) * dx)
df.list_timings(df.TimingClear.clear, [df.TimingType.wall])
dJdu = compute_gradient(J, Control(ctrl))
df.list_timings(df.TimingClear.clear, [df.TimingType.wall])

import matplotlib.pyplot as plt

u = Complex(u[0], u[1])

u = project(u, Vre)

plt.clf()
plot(u.imag)
plt.show()
plt.figure()
plot(dJdu)
plt.show()
