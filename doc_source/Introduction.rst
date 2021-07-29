.. _Introduction:

Introduction
============
    **XModFit** stands for **X**\-ray **Mod**\eling and **Fit**\ting. **XModFit** provides a graphical user interface (GUI) platform to quickly write fitting functions for any 1-dimensional data primarily focussed on X-ray scattering. The package is developed at `NSF's ChemMatCARS (Sector 15) <https://chemmatcars.uchicago.edu/>`_  at `Advanced Photon Source <https://www.aps.anl.gov/>`_ , USA.

    If you are using any components of **XModFit** for your research/work please do not forget to acknowledge (see :ref:`Acknowledgements`).

    **XModFit** provides a platform to simulate and fit a model to 1D data. It uses `LMFIT <https://lmfit.github.io/lmfit-py/>`_ python library for fitting. Some of the commonly used functions are provided under different categories and the users can develop their own categories and fitting functions by using an easy to use template within a :ref:`Function_Editor`


.. figure:: ./Figures/XModFit_in_Action.png
    :figwidth: 100%

    **XModFit** in action in which a Small Angle X-ray Scattering data is fitted with poly-dispersed **Sphere** model with **Log-Normal** distribution.

Features
********

    1. Read and fit multiple data sets
    2. Functions are categorized under different types and experimental techniques
    3. Easy to add new categories and new functions within the categories
    4. Once the function is defined properly all the free and fitting parameters will be available within the GUI as tables.
    5. An in-built :ref:`Function_Editor` is provided with an easy to use template.
    6. A :ref:`Data_Dialog` is provided for importing and manipulating data files.
    7. Another cool feature of :ref:`XAnoS_Fit` is the ability to view and save other functions/parameters generated during the calculation/evaluation of a user supplied functions.


Usage
*****

    :ref:`XModFit` can be used as stand-alone python fitting package by running this in terminal::

        python xmodfit.py

    The widget can be used as a widget with any other python application.
