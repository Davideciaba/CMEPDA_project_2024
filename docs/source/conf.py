# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CMEPDA_PROJECT_2024'
copyright = '2026, Tancredi Lipari & Davide Ciabattoni'
author = 'Tancredi Lipari & Davide Ciabattoni'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

import os
import sys

# 1. Path resolution: punta alla cartella Python del tuo progetto
sys.path.insert(0, os.path.abspath('../../Python'))

project = 'CMEPDA_project_2024'
copyright = '2024, Authors'
author = 'Authors'
release = '1.0'

# 2. Extensions: autodoc per leggere il codice, napoleon per lo stile Google, viewcode per il link al sorgente
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Impostazioni di default per autodoc (inclusione di costruttori privati se documentati)
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# 3. HTML Theme: utilizzo del layout standard di ReadTheDocs
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']