
Gyptis
======

:Release: |release|
:Date: |today|

Gyptis is an open-source Python package to solve problems 
in Photonics and Electromagnetism using the Finite Element method. 
It relies on Gmsh_ for geometry definition and meshing 
and Fenics_ for solving Maxwell's equations.


.. _Gmsh: https://gmsh.info/
.. _Fenics: https://fenicsproject.org/


.. .. toctree::
..    :maxdepth: 1
..
..    install_upgrade
..    api
..    release

Examples
--------

Tutorials with worked examples and background information for
submodules.

.. toctree::
   :maxdepth: 2

   auto_examples/index.rst




API Reference
-------------

:mod:`gyptis.geometry`: Defining geometry and generating meshes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: gyptis.geometry
    :no-members:
    :no-inherited-members:

Base classes
""""""""""""
.. currentmodule:: gyptis

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst
   
   geometry.Model
   geometry.BoxPML



-------------------------------------



:mod:`gyptis.complex`: Complex numbers support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: gyptis.complex
    :no-members:
    :no-inherited-members:

Base classes
""""""""""""
.. currentmodule:: gyptis

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class_with_call.rst
   
   complex.Complex


Functions
"""""""""
.. currentmodule:: gyptis

.. autosummary::
  :nosignatures:
  :toctree: generated/
  :template: function.rst
  
  complex.iscomplex


-------------------------------------


Indices and search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
