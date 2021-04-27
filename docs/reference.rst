


=============
Documentation
=============

.. only:: html

  .. raw:: html

    <div class=badges>

  .. include:: ../README.rst
    :end-before: inclusion-marker-badges
    
  .. raw:: html

    </div>
  


Welcome! This is the documentation for Gyptis |release|, last updated on |today|.

Gyptis is an open-source Python package to solve problems 
in Photonics and Electromagnetism using the Finite Element method. 
It relies on Gmsh_ for geometry definition and meshing 
and FEniCS_ for solving Maxwell's equations.

For a more practical introduction, check the 
:ref:`tutorials with worked examples <sphx_glr_auto_examples>`.

A pdf version of this documentation is available to download `here <https://gyptis.gitlab.io/_downloads/gyptis.pdf>`_.

.. only:: html

  You can use the search bar on the left or the :ref:`search page<search>` to look up for something 
  in the documentation. Also useful are the :ref:`index<genindex>` 
  and the :ref:`module index<modindex>` where of objects and modules are listed.

.. toctree::
   :maxdepth: 2
   :hidden:
   
   installation
   api



.. _Gmsh: https://gmsh.info/
.. _FEniCS: https://fenicsproject.org/

.. bibliography::

.. 
.. 
.. Lists and quote-like blocks
.. ---------------------------
.. 
.. * Bulleted list
.. * with two items
.. 
.. #. Numbered list
.. #. with
.. #. three items
.. 
.. * Nested
.. 
..  #. List
.. 
..    * Hooray
..    * Hooray
..    * Hooray
..    * Hooray
.. 
..  #. List
.. 
..    * Hooray
..    * Hooray
..    * Hooray
..    * Hooray
.. 
.. * Nested
.. 
..  #. List
.. 
.. 
.. 
.. term (up to a line of text)
..   Definition of the term, which must be indented
.. 
..   and can even consist of multiple paragraphs
.. 
.. next term
..   Description.
.. 
.. 
.. 
.. Paragraph heading
.. """""""""""""""""
.. 
.. 
.. 
.. .. topic:: Topic
.. 
..  A topic is like a block quote with a title, or a self-contained section with
..  no subsections. Use the "topic" directive to indicate a self-contained idea
..  that is separate from the flow of the document. Topics may occur anywhere a
..  section or transition may occur. Body elements and topics may not contain
..  nested topics.
.. 
.. .. parsed-literal::
.. 
..  parsed-literal is a literal-looking block with **parsed** *text*
.. 
.. Epigraph
.. ^^^^^^^^
.. 
.. .. epigraph::
.. 
..  No matter where you go, there you are.
.. 
..  -- Buckaroo Banzai
.. 
.. Compound paragraph
.. ^^^^^^^^^^^^^^^^^^
.. 
.. .. compound::
.. 
..   This is a compound paragraph. The 'rm' command is very dangerous.  If you
..   are logged in as root and enter ::
.. 
..       cd /
..       rm -rf *
.. 
..   you will erase the entire contents of your file system.
.. 
.. 
.. 
..   Raw HTML
..   ^^^^^^^^
.. 
..   .. raw:: html
.. 
..     <span style="color: red;">This is some raw HTML.</span>
.. 
.. Sidebar
.. -------
.. 
.. .. sidebar:: Sidebar
.. 
..   Sidebars are like miniature, parallel documents that occur inside other
..   documents, providing related or reference material. A sidebar is typically
..   offset by a border and "floats" to the side of the page; the document's main
..   text may flow around it. Sidebars can also be likened to super-footnotes;
..   their content is outside of the flow of the document's main text.
.. 
.. Admonition blocks
.. -----------------
.. 
.. .. attention:: attention block block block block block block block block block
..     block block
.. 
.. .. caution:: caution block
.. 
.. .. danger:: danger block
.. 
.. .. error:: error block
.. 
.. .. hint:: hint block
.. 
.. .. important:: important block
.. 
.. .. note:: note block
.. 
.. .. tip:: tip block
.. 
.. .. warning:: warning block
.. 
.. .. admonition:: custom admonition
.. 
..   with content
