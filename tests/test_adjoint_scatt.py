# from df import *
# from df_adjoint import *
import dolfin as df

import gyptis
from gyptis.complex import *
from gyptis.geometry import *
from gyptis.helpers import *
from gyptis.helpers import _get_form
from gyptis.materials import *
from gyptis.optimize import *
from gyptis.physics import *
from gyptis.plotting import *

plt.ion()


pi = np.pi

parmesh = 5
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
submesh = dolfin.SubMesh(mesh, markers, domains["lens"])
Actrl = df.FunctionSpace(mesh, "DG", 0)
ASub = dolfin.FunctionSpace(submesh, "DG", 0)
ctrl0 = df.Expression("0.5", degree=2)
ctrl = project(ctrl0, Actrl)
eps_lens_func = ctrl * (eps_max - eps_min) + eps_min
eps_lens_func *= Complex(1, 0)


epsilon = dict(lens=eps_lens_func, box=eps_box, target=eps_box)
mu = dict(lens=1, box=1, target=1)

# if __name__ == "__main__":
s = Scatt2D(geom, epsilon, mu, degree=2, lambda0=lambda0, theta0=pi)

xsub = project(ctrl, ASub)
x = function2array(xsub)
nvar = len(x)

# x = np.random.rand(nvar)


rfilt = 0.1


def simulation(x, proj_level=0, filt=True, proj=True, plot_optim=True):
    x = array2function(x, ASub)
    if filt:
        x = filtering(x, rfilt, solver="iterative")
    if proj:
        x = projection(x, beta=2 ** proj_level)
    x = project(x, ASub)
    x = function2array(x)

    a = dolfin.Function(Actrl)
    m = markers.where_equal(domains["lens"])
    a = function2array(a)
    a[m] = x
    ctrl = array2function(a, Actrl)

    eps_lens_func = ctrl * (eps_max - eps_min) + eps_min
    eps_lens_func *= Complex(1, 0)

    s.epsilon["lens"] = eps_lens_func

    s.prepare()
    s.weak_form()
    s.assemble()
    s.build_system()
    s.solve()

    field = s.u + s.u0
    J = -assemble(inner(field, field.conj) * s.dx("target")).real / Starget
    if gyptis.ADJOINT:
        dJdx = dolfin.compute_gradient(J, dolfin.Control(ctrl))

        dJdx = project(dJdx, ASub)
        dJdx = function2array(dJdx)
    else:
        dJdx = None

    if plot_optim:

        ax.clear()
        field = project(field.module, s.real_space)
        _, CB = plot(field, ax=ax)
        plt.plot(xtarget, ytarget, "og")
        CB.remove()
        ctrl_plt = project(ctrl, ASub)
        dolfin.plot(
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
        J, grad = simulation(x, plot_optim=False)


# J, grad = simulation(x)

if __name__ == "__main__":

    # fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    from scipy.optimize import minimize

    maxiter = 5
    bounds = [(0, 1) for i in range(nvar)]
    for proj_level in range(8):
        print("projection level: ", proj_level)
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

    x = array2function(x, ASub)
    x = filtering(x, rfilt, solver="iterative")
    x = projection(x, beta=2 ** proj_level)
    x = project(x, ASub)
    x = function2array(x)
    x[x < 0.5] = 0
    x[x >= 0.5] = 1
    J, grad = simulation(x, proj_level=16, proj=False, filt=False)

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
