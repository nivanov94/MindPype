# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.append(os.path.abspath(
    os.path.join(__file__, "../bcipy")
))

project = 'bcipy'
copyright = '2022, Nicolas Ivanov, Aaron Lio'
author = 'Nicolas Ivanov, Aaron Lio'

version = '0.1'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.githubpages'
]

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

source_suffix = ['.rst']
master_doc = 'index'
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True
napoleon_strip_signature_backslash = True
autodoc_strip_signature_backslash = True
strip_signature_backslash = True

pygments_style = 'sphinx'

autodoc_mock_imports = ['matplotlib', 'more-itertools',
                        'numpy', 'numpydoc', 'pyriemann', 
                        'scikit-learn', 'scipy', 'pylsl', 'pyxdf', 'sklearn', 'mpl_toolkits', 'liesl', 'mne']
