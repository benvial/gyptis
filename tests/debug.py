#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import matplotlib.pyplot as plt
from fenics import *
from fenics_adjoint import *

mesh = UnitIntervalMesh(200)
n = FacetNormal(mesh)


def boundary(x, on_boundary):
    return on_boundary


P = FiniteElement("CG", interval, 2)
element = MixedElement([P, P, P])
T = FunctionSpace(mesh, element)

u_test, k_test, w_test = TestFunctions(T)
u, k, w = TrialFunctions(T)

T_n = Function(T)
u_n, k_n, w_n = split(T_n)
T_ = project(Expression(("0.0", "1e-7", "32000"), degree=1), T)
u_, k_, w_ = split(T_)

gradP = Constant(10.0)
mu = Constant(1e-4)

V = FunctionSpace(mesh, "CG", 2)
f = project(Expression("1.0", name="Control", degree=1), V)
control = Control(f)

bc_u = DirichletBC(T.sub(0), Constant(0.0), boundary)
bc_k = DirichletBC(T.sub(1), Constant(0.0), boundary)
bc_w = DirichletBC(T.sub(2), Constant(32000), boundary)
bc = [bc_u, bc_k, bc_w]

F1 = dot(grad(u), grad(u_test)) * dx - gradP * u_test * dx
F2 = (
    dot(dot(grad(u_), grad(u_)), k_test) * dx
    - mu * dot(grad(k), grad(k_test)) * dx
    + mu * dot(grad(k), (k_test * n)) * ds
)
F3 = (
    f * w_test * dx
    + dot(dot(grad(u_), grad(u_)), w_test) * dx
    - mu * dot(grad(w), grad(w_test)) * dx
    + mu * dot(grad(w), (w_test * n)) * ds
)

F = F1 + F2 + F3

a = lhs(F)
L = rhs(F)
#
# for i in range(10):
#
#     solve(a == L, T_n, bc)
#     # T_.assign(project(Expression(("0.6*u_+0.4*u_n","0.6*k_+0.4*k_n","0.6*w_+0.4*w_n"), degree=1, u_n=T_n.sub(0), k_n=T_n.sub(1), w_n=T_n.sub(2), u_=T_.sub(0), k_=T_.sub(1), w_=T_.sub(2)), T))
#     tu_n, tk_n, tw_n = T_n.split()
#     tu_, tk_, tw_ = T_.split()
#     T_.assign(project(as_vector((0.6*tu_+0.4*tu_n,0.6*tk_+0.4*tk_n,0.6*tw_+0.4*tw_n)), T))

solve(a == L, T_n, bc)

tu_, tk_, tw_ = T_.split()
tu_n, tk_n, tw_n = T_n.split()
T_.assign(
    project(
        as_vector(
            (0.6 * tu_ + 0.4 * tu_n, 0.6 * tk_ + 0.4 * tk_n, 0.6 * tw_ + 0.4 * tw_n)
        ),
        T,
    )
)


plot(u_)
plt.show()

J = assemble(inner(u_, u_) * dx)
dJdEpsilon = compute_gradient(J, control)


plot(dJdEpsilon)
plt.show()
