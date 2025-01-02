# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TFcomb'
copyright = '2025, Chen Li'
author = 'Chen Li'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# 添加 'nbsphinx' 到 extensions 列表
extensions = [
    'nbsphinx',
    'sphinx.ext.mathjax',  # 如果你的 notebook 包含数学公式
    'sphinx.ext.viewcode', # 可选，用于显示源码
    # 其他扩展...
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    
    'sphinx.ext.napoleon',      # 如果你使用 Google 或 NumPy 风格注释，则需要这个
]

# 配置 nbsphinx 的选项（可选）
nbsphinx_execute = 'never'  # 自动执行 notebook，可选值: 'always', 'never', 'auto'
nbsphinx_allow_errors = False  # 如果 notebook 中有错误，构建时是否允许
nbsphinx_timeout = 60  # 执行 notebook 的超时时间（秒）

import os
import sys
# sys.path.insert(0, os.path.abspath('../../code_test'))

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# html_theme_options = {
#     'collapse_navigation': True,  # 启用折叠功能
#     'navigation_depth': 3,        # 设置显示的最大层级
#     'sticky_navigation': True,    # 保持侧边栏可见
# }
