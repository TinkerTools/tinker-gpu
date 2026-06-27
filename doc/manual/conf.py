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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import datetime
project = "Tinker-GPU User Manual"
html_title = project
author_html = "Zhi Wang & Jay W. Ponder"
author_latex = "Zhi Wang \& Jay W. Ponder"
copyright = '%s, %s' % (datetime.datetime.now().year, author_html)
master_doc = 'index'
latex_documents = [(master_doc, 'tinker-gpu-manual.tex', project,
     author_latex,
    'manual')]

numfig = True

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinxcontrib.bibtex'
]
bibtex_bibfiles = ['m/refs.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/tinkertools/tinker-gpu',
    'twitter_url': 'https://twitter.com/tinkertoolsmd'
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for LaTeXPDF output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
    'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
    'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
    'preamble': r'''
%\usepackage{charter}
%\usepackage{fouriernc}
\usepackage[notextcomp]{kpfonts}
% sans serif
\usepackage[defaultsans]{lato}
% monospace
\usepackage{inconsolata}

\usepackage[normalem]{ulem} % strikethrough text: \sout{text}

% \usepackage{geometry} % already used
\geometry{paperheight=8.5in,paperwidth=5.5in,top=1.0in,bottom=1.0in,left=0.5in,right=0.5in,heightrounded}

\addto{\captionsenglish}{\renewcommand{\bibname}{References}}

% prevent hyphenation
\hyphenation{LPISTON}
''',

   'sphinxsetup': '''
       TitleColor={rgb}{0,0,0},
       InnerLinkColor={rgb}{0,0,0},
       OuterLinkColor={rgb}{0,0,0},
       VerbatimColor={rgb}{0.9,0.9,0.9},
       VerbatimBorderColor={rgb}{1,1,1}
'''
}

from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.plugin import register_plugin
from pybtex.style.template import sentence, optional, words

class UnsrtStyleModified(UnsrtStyle):
    def format_web_refs(self, e):
        if 'doi' in e.fields:
            return sentence [ optional [ self.format_doi(e) ] ]
        elif 'pubmed' in e.fields:
            return sentence [ optional [ self.format_pubmed(e) ] ]
        elif 'url' in e.fields:
            return sentence [ optional [ self.format_url(e) ] ]
        else:
            return words ['']

register_plugin('pybtex.style.formatting', 'unsrt-modified', UnsrtStyleModified)
