# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "ActionWeaver"
copyright = '2023, Teng "Niel" Hu'
author = 'Teng "Niel" Hu'
release = "0.0.9"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "m2r2",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

source_suffix = [".rst", ".md"]
# note: do not add .ipynb when nbspinx is enabled, otherwise you get the "missing title" error

templates_path = ["_templates"]

# Exclude build directory and Jupyter backup files:
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
