import numpy as np
from pytheas import Periodic3D, genmat

fem = Periodic3D()
fem.rm_tmp_dir()
fem.lambda0 = 1.1
fem.theta_deg = 0
fem.phi_deg = 0
fem.psi_deg = 0
fem.period_x = 1.0  #: flt: periodicity in x-direction
fem.period_y = 1.0  #: flt: periodicity in y-direction
fem.thick_L1 = 1
fem.thick_L2 = 0.1
fem.thick_L3 = 0.5
fem.thick_L4 = 0.1
fem.thick_L5 = 0.1
fem.thick_L6 = 1
fem.PML_top = 1.1  #: flt: thickness pml top
fem.PML_bot = 1.1  #: flt: thickness pml bot
fem.eps_L2 = 1 - 0.0 * 1j  #: flt: permittivity layer 2
fem.eps_L3 = 1 - 0.1 * 1j  #: flt: permittivity layer 3
fem.eps_L4 = 1 - 0.0 * 1j  #: flt: permittivity layer 4
fem.eps_L5 = 1 - 0.0 * 1j  #: flt: permittivity layer 5


fem.eps_L1 = 1 - 0.0 * 1j  #: flt: permittivity superstrate
fem.eps_L6 = 3 - 0.0 * 1j  #: flt: permittivity substrate

fem.parmesh_des = 12
fem.parmesh = 6
fem.parmesh_pml = fem.parmesh * 2 / 3
fem.N_d_order = 0
fem.el_order = 1  #: int: order of basis function (1 or 2)
fem.matprop_pattern = [3 ** 0.5, 1]  # refractive index values

fem.getdp_verbose = 4
fem.gmsh_verbose = 4
fem.python_verbose = 1
fem.initialize()

layer_diopter = fem.ancillary_problem()

dasd

fem.make_mesh()


xsym = True
ysym = True
threeD = False
genmat.np.random.seed(100)
mat = genmat.MaterialDensity()  # instanciate
mat.n_x, mat.n_y, mat.n_z = 2 ** 8, 2 ** 8, 1  # sizes
if threeD:
    mat.n_z = mat.n_x
mat.xsym = xsym  # symmetric with respect to x?
mat.ysym = ysym  # symmetric with respect to y?
mat.p_seed = mat.mat_rand  # fix the pattern random seed
mat.nb_threshold = 2  # number of materials
mat._threshold_val = np.random.permutation(mat.threshold_val)
mat.pattern = mat.discrete_pattern


import matplotlib.pyplot as plt

plt.ion()
q = np.zeros_like(mat.pattern)

x, y, z = mat.mat_grid
x = x / (len(x) - 1)
y = y / (len(y) - 1)


q[x < 0.25] = 1
q[x > 0.75] = 1
q[y < 0.25] = 1
q[y > 0.75] = 1

plt.imshow(q[:, :, 0])
mat.pattern = q

fem.register_pattern(mat.pattern, mat._threshold_val)
fem.compute_solution()
effs = fem.diffraction_efficiencies()
fem.postpro_fields_pos()
fem.open_gmsh_gui()
print("effs = ", effs)
