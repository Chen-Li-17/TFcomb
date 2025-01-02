import os
import shutil

# 复制上上级目录中的 notebooks 到构建目录
source_notebooks = os.path.abspath('../../tutorial_example')
target_notebooks = os.path.join(os.path.abspath('.'), 'tutorial_example')

if not os.path.exists(target_notebooks):
    shutil.copytree(source_notebooks, target_notebooks)
