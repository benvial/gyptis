# -*- coding: utf-8 -*-
"""
3D Grating
==========

An example of a bi-periodic diffraction grating.
"""

from gyptis.grating_3d import *

##############################################################################
# The diffracted field :math:`{\mathbf E}^d` can be decomposed in a Rayley series:
#
# .. math::
#
#    {\mathbf E}^d(x,y,z) = \sum_{(n,m) \in \mathbb Z^2}
#    {\mathbf U}_{nm}(z) e^{-i(\alpha_n x + \beta_m y)}
# with :math:`\alpha_n=\alpha_0 + p_n`, :math:`\beta_m=\beta_0 + q_m`,
# :math:`p_n=2\pi n/d_x` and :math:`q_m=2\pi m/d_y`.
#
# The coefficients of the decomposition can be expressed as:
#
# .. math::
#
#    {\mathbf U}_{nm}(z) = \frac{1}{d_x d_y}\int_{-d_x/2}^{d_x/2}\int_{-d_y/2}^{d_y/2}
#    {\mathbf E}^d(x,y,z) e^{-i(\alpha_n x + \beta_m y)}\mathrm d x  \mathrm d y
#
# Note that we solve for the periodic part of the total field
# :math:`{\mathbf E}_\#^d = {\mathbf E}^d e^{-i(\alpha_n x + \beta_m y)}`.
#
# In the substrate (-), we have :math:`{\mathbf U}_{nm}(z) = {\mathbf V}^{-}_{nm} e^{-i\gamma^{-}_{nm}}`
# and in the superstrate (+), :math:`{\mathbf U}_{nm}(z) = {\mathbf V}^{+}_{nm} e^{i\gamma^{+}_{nm}}`
#
# The total diffracted field is
