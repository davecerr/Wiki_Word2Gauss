
import numpy as np

from setuptools  import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension(
        "word2gauss.embeddings",
        sources=['word2gauss/embeddings.pyx'],
        include_dirs=[np.get_include()],
        language="c++",
        # extra_compile_args=["-fopenmp", "-ftree-vectorizer-verbose=2", "-O3", "-ffast-math"],
        # extra_link_args=["-fopenmp"]
    )
]


setup(name='word2gauss',
    packages=['word2gauss'],
    package_dir={'word2gauss': 'word2gauss'},
    cmdclass         = {'build_ext': build_ext},
    ext_modules      = ext_modules,
)
