# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from recommonmark.transform import AutoStructify

sys.path.insert(0, os.path.abspath("."))
import qibolab

# -- Project information -----------------------------------------------------

project = "qibolab"
copyright = "2021, The Qibo team"
author = "The Qibo team"

release = qibolab.__version__


# -- General configuration ---------------------------------------------------

# https://stackoverflow.com/questions/56336234/build-fail-sphinx-error-contents-rst-not-found
# master_doc = "index"

autodoc_mock_imports = ["qm"]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "recommonmark",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "furo"
html_favicon = "favicon.ico"

# custom title
html_title = "Qibolab · v" + release

html_theme_options = {
    "light_logo": "qibo_logo_dark.svg",
    "dark_logo": "qibo_logo_light.svg",
    "light_css_variables": {
        "color-brand-primary": "#6400FF",
        "color-brand-secondary": "#6400FF",
        "color-brand-content": "#6400FF",
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Intersphinx  -------------------------------------------------------------

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}


# -- Doctest ------------------------------------------------------------------
#

doctest_path = [os.path.abspath("../examples")]

# -- Autodoc ------------------------------------------------------------------
#
autodoc_member_order = "bysource"


# Adapted this from
# https://github.com/readthedocs/recommonmark/blob/ddd56e7717e9745f11300059e4268e204138a6b1/docs/conf.py
# app setup hook


def run_apidoc(_):
    """Extract autodoc directives from package structure."""
    source = Path(__file__).parent
    docs_dest = source / "api-reference"
    package = source.parents[1] / "src" / "qibolab"
    apidoc.main(["--module-first --no-toc", "-o", str(docs_dest), str(package)])


def setup(app):
    app.add_config_value("recommonmark_config", {"enable_eval_rst": True}, True)
    app.add_transform(AutoStructify)
    app.add_css_file("css/style.css")

    # app.connect("builder-inited", run_apidoc)


html_show_sourcelink = False
