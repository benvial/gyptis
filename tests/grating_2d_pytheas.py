from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from pytheas import Periodic2D, genmat

##############################################################################
# Then we need to instanciate the class :py:class:`Periodic2D`:

fem = Periodic2D()

fem.d = 20  #: flt: period
fem.h_sup = 30  #: flt: "thickness" superstrate
fem.h_sub = 40  #: flt: "thickness" substrate
fem.h_layer1 = 10  #: flt: thickness layer 1
fem.h_layer2 = 15  #: flt: thickness layer 2
fem.h_des = 5  #: flt: thickness layer design
fem.h_pmltop = 40  #: flt: thickness pml top
fem.h_pmlbot = 40  #: flt: thickness pml bot
fem.a_pml = 1  #: flt: PMLs parameter, real part
fem.b_pml = 1  #: flt: PMLs parameter, imaginary part
fem.eps_sup = 1  #: flt: permittivity superstrate
fem.eps_sub = 5  #: flt: permittivity substrate
fem.eps_layer1 = 3 - 0.1j  #: flt: permittivity layer 1
fem.eps_layer2 = 1  #: flt: permittivity layer 2
fem.eps_des = 1  #: flt: permittivity layer design
fem.lambda0 = 40  #: flt: incident wavelength
fem.theta_deg = 30.0  #: flt: incident angle
fem.pola = "TE"  #: str: polarization (TE or TM)
fem.lambda_mesh = fem.lambda0  #: flt: incident wavelength
#: mesh parameters, correspond to a mesh size of lambda_mesh/(n*parmesh),
#: where n is the refractive index of the medium
fem.parmesh_des = 40
fem.parmesh = 40
fem.parmesh_pml = fem.parmesh * 2 / 3
fem.type_des = "elements"
fem.quad_mesh_flag = True


eps_island = 6 - 1j


fem.getdp_verbose = 0
fem.gmsh_verbose = 0
fem.python_verbose = 0

fem.initialize()
mesh = fem.make_mesh()

genmat.np.random.seed(100)
mat = genmat.MaterialDensity()  # instanciate
mat.n_x, mat.n_y, mat.n_z = 2 ** 7, 2 ** 7, 1  # sizes
mat.xsym = True  # symmetric with respect to x?
mat.p_seed = mat.mat_rand  # fix the pattern random seed
mat.nb_threshold = 2  # number of materials


pattern = np.zeros_like(mat.discrete_pattern)

x, y, z = mat.mat_grid
x = x / (len(x) - 1)
y = y / (len(y) - 1)

w = 0.5 * 10 / fem.d
pattern[abs(x - 0.5) > w] = 1
# pattern[x>0.75 ] = 1
# pattern[y<0.25] = 1
# pattern[y>0.75 ] = 1
#
# import matplotlib.pyplot as plt
# plt.ion()
# c = plt.imshow(pattern[:,:,0])
# plt.colorbar(c)


fem.register_pattern(pattern, mat._threshold_val)
fem.matprop_pattern = [(eps_island) ** 0.5, 1]  # refractive index values

fem.compute_solution()

effs = fem.diffraction_efficiencies(orders=True)
print("efficiencies")
pprint(effs)
# fem.postpro_fields_pos()

# fem.open_gmsh_gui()

fem.pola = "TM"
fem.compute_solution()

effs = fem.diffraction_efficiencies(orders=True)
print("efficiencies")
pprint(effs)
