# from df import *
# from df_adjoint import *


import time

import gyptis
from gyptis import dolfin as df
from gyptis.complex import *
from gyptis.geometry import *
from gyptis.helpers import *
from gyptis.materials import *
from gyptis.optimize import *
from gyptis.physics import *
from gyptis.plotting import *

plt.ion()


np.random.seed(123456)

pi = np.pi

parmesh = 4
parmesh_lens = parmesh * 2.0
lambda0 = 1

eps_min, eps_max = 1, 6
eps_box = 1
lx, ly = 1, 4
rtarget = lambda0 / 5
xtarget, ytarget = lx * 2, 0
Starget = pi * rtarget ** 2


geom = BoxPML(dim=2, box_size=(lx + lambda0 * 4, ly + lambda0 * 1), pml_width=(1, 1))

lens = geom.add_rectangle(-lx / 2, -ly / 2, 0, lx, ly)
lens, geom.box = geom.fragment(lens, geom.box)
# target = geom.add_rectangle(xtarget, -ly/2, 0, lx/5, ly)
target = geom.add_disk(xtarget, ytarget, 0, rtarget, rtarget)
target, geom.box = geom.fragment(target, geom.box)
geom.add_physical(geom.box, "box")
geom.add_physical(lens, "lens")
geom.add_physical(target, "target")
pmls = [d for d in geom.subdomains["surfaces"] if d.startswith("pml")]
geom.set_size(pmls, lambda0 / parmesh * 0.7)
geom.set_size("box", lambda0 / (parmesh))
geom.set_size("target", lambda0 / (parmesh))
geom.set_size("lens", lambda0 / (parmesh_lens * eps_max ** 0.5))

geom.build(interactive=False)

mesh = geom.mesh_object["mesh"]
markers = geom.mesh_object["markers"]["triangle"]
domains = geom.subdomains["surfaces"]
submesh = df.SubMesh(mesh, markers, domains["lens"])
Actrl = df.FunctionSpace(mesh, "DG", 0)
Asub = df.FunctionSpace(submesh, "DG", 0)
ctrl0 = df.Expression("0.5", degree=2)
ctrl = project(ctrl0, Actrl)
eps_lens_func = ctrl * (eps_max - eps_min) + eps_min
eps_lens_func *= Complex(1, 0)


epsilon = dict(lens=eps_lens_func, box=eps_box, target=eps_box)
mu = dict(lens=1, box=1, target=1)

# if __name__ == "__main__":
s = Scatt2D(geom, epsilon, mu, degree=2, lambda0=lambda0, theta0=pi)

nvar = Asub.dim()
# nvar =  Actrl.dim()

x = np.random.rand(nvar)

plt.close("all")
rfilt = 0.1

ja = 0


def simulation(x, proj_level=0, filt=True, proj=True, plot_optim=True, reset=True):
    global ja

    if gyptis.ADJOINT and reset:
        df.set_working_tape(df.Tape())

    x = array2function(x, Asub)
    if filt:
        x = filtering(x, rfilt, solver="iterative")
    if proj:
        x = projection(x, beta=2 ** proj_level)
    x = project(x, Asub)
    x = function2array(x)

    a = df.Function(Actrl)
    m = markers.where_equal(domains["lens"])
    a = function2array(a)
    a[m] = x
    ctrl = array2function(a, Actrl)

    eps_lens_func = ctrl * (eps_max - eps_min) + eps_min
    eps_lens_func *= Complex(1, 0)

    s.epsilon["lens"] = eps_lens_func

    ttot = -time.time()
    s.prepare()
    s.weak_form()
    if ja == 0:
        s.assemble()
    else:
        s.assemble(["lens"], ["lens"], [])
    s.build_system()
    t = -time.time()
    # again = False if ja==0 else True
    s.solve()
    t += time.time()
    ttot += time.time()
    print("direct: ", t)
    print("direct (total): ", ttot)
    ja += 1

    field = s.u + s.u0
    J = -assemble(inner(field, field.conj) * s.dx("target")).real / Starget

    # target_phi = 0
    #
    # J += assemble(abs(field.phase - target_phi) * 10 * s.dx("target")) / Starget
    #
    #
    # phase_tar=df.project(field.phase,s.real_space)
    # print(phase_tar(xtarget,ytarget), target_phi)
    #
    # mod_tar=df.project(field.module,s.real_space)
    # print(mod_tar(xtarget,ytarget))
    #
    print("   >>> objective = ", J)
    if gyptis.ADJOINT:
        t = -time.time()
        dJdx = df.compute_gradient(J, df.Control(ctrl))
        t += time.time()
        print("adjoint: ", t)

        dJdx = project(dJdx, Asub)
        dJdx = function2array(dJdx)
    else:
        dJdx = None

    if plot_optim:

        ax.clear()
        field = project(field.module, s.real_space)
        _, CB = plot(field, ax=ax)
        plt.plot(xtarget, ytarget, "og")
        CB.remove()
        ctrl_plt = project(ctrl, Asub)
        df.plot(
            ctrl_plt,
            cmap="binary",
            alpha=0.4,
            lw=0.00,
            edgecolor="face",
            vmin=0,
            vmax=1,
        )

        b = geom.box_size
        ax.set_xlim(-b[0] / 2, b[0] / 2)
        ax.set_ylim(-b[1] / 2, b[1] / 2)
        plt.axis("off")
        plt.tight_layout()
        plt.pause(0.1)

    return J, dJdx


def test_simu():
    for s.polarization in ["TE", "TM"]:
        J, grad = simulation(x, plot_optim=False, reset=True)


def test_taylor():
    if gyptis.ADJOINT:

        def check_taylor_test(s):
            for s.polarization in ["TE", "TM"]:
                df.set_working_tape(df.Tape())
                s.epsilon["lens"] = eps_lens_func
                s.prepare()
                s.weak_form()
                s.assemble()
                s.build_system()
                s.solve()
                h = df.Function(Actrl)
                h.vector()[:] = 1e-2 * np.random.rand(Actrl.dim())
                field = s.u + s.u0
                J = -assemble(inner(field, field.conj) * s.dx("target")).real / Starget
                Jhat = df.ReducedFunctional(
                    J, df.Control(ctrl)
                )  # the functional as a pure function of nu
                conv_rate = df.taylor_test(Jhat, ctrl, h)
                print("convergence rate = ", conv_rate)
                assert abs(conv_rate - 2) < 1e-2

        check_taylor_test(s)

        s.source = "LS"
        s.xs = -xtarget, -ytarget
        check_taylor_test(s)


# J, grad = simulation(x)

if __name__ == "__main__":

    # fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    from scipy.optimize import minimize

    maxiter = 15
    bounds = [(0, 1) for i in range(nvar)]
    for proj_level in range(8):
        print("\n#########################")
        print("projection level: ", proj_level)
        print("#########################\n")
        opt = minimize(
            simulation,
            x,
            args=(proj_level),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        x = opt.x

    x = array2function(x, Asub)
    x = filtering(x, rfilt, solver="iterative")
    x = projection(x, beta=2 ** proj_level)
    x = project(x, Asub)
    x = function2array(x)
    x[x < 0.5] = 0
    x[x >= 0.5] = 1
    J, grad = simulation(x, proj=False, filt=False)

    ###### test perfs
    s.prepare()
    s.weak_form()
    s.assemble()
    s.build_system()

    VVect = df.VectorFunctionSpace(s.mesh, s.element, s.degree)
    u = df.Function(VVect)

    t = -time.time()
    solver = df.LUSolver("mumps")
    solver.solve(s.matrix, u.vector(), s.vector)
    t += time.time()
    print("direct: ", t)

    t = -time.time()
    solver = df.LUSolver(s.matrix, "mumps")
    solver.solve(u.vector(), s.vector)
    t += time.time()
    print("direct: ", t)

    t = -time.time()
    solver.solve(u.vector(), s.vector)
    t += time.time()
    print("direct: ", t)

    s.u = Complex(*u.split())

    field = s.u + s.u0
    J = -assemble(inner(field, field.conj) * s.dx("target")).real / Starget

    if gyptis.ADJOINT:
        t = -time.time()
        dJdx = df.compute_gradient(J, df.Control(ctrl))
        t += time.time()
        print("adjoint: ", t)

    #


# s.solve()
#
# # u = self.u
# u = Complex(*s.u.split())
# J = assemble(inner(u, u.conj) * s.dx).real
#
# u = project(u,s.real_space)
#
# try:
#     dJdu = df.compute_gradient(J, df.Control(ctrl))
# except:
#
#     dJdu = df.compute_gradient(J, df.Control(ctrl))
#
#
# # plt.close("all")
# # plt.ion()
# # plotcplx(u)
# # plt.show()
#
#
# ttot = []
# vals =[]
# from time import time as tm
# for d, f in s.lhs.items():
#     print(d)
#     for i in f:
#         t = -tm()
#         a = assemble(i*s.dx(d))
#         t += tm()
#         print(t)
#         ttot.append(t)
#         vals.append(a)
#
# print(sum(ttot))
#
#
# forms = list(s.lhs.values())
#
# ff = sum(forms[1:], start=forms[0])
# F = sum(ff[1:], start=ff[0])
#
#
#
# t = -tm()
# assemble(F*s.dx)
# t += tm()
# print(t)
#
#
# forms = []
#
# for d, f in s.Ah.items():
#     print(d)
#     for g in f:
#         # print(g.form)
#         forms.append(g.form)
#
#
# ff = sum(forms[1:], start=forms[0])
#
#
# a.form.integrals()[0] * k0**2
