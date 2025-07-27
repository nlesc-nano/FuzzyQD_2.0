import os
import sys
import subprocess
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake is required to build the C++ extension.")

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Release'
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        # Prefer Ninja if available (fast builds)
        generator = []
        if shutil.which("ninja"):
            generator = ['-G', 'Ninja']

        # Set up CMake args for conda/mamba compilers
        cmake_args = [
            ext.sourcedir,
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
            f'-DPython3_EXECUTABLE={sys.executable}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',  # legacy
        ]

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            cmake_args.append(f'-DCMAKE_PREFIX_PATH={conda_prefix}')

        # Support CMAKE_ARGS injected by conda compilers/metapackage
        extra = os.environ.get("CMAKE_ARGS")
        if extra:
            cmake_args += extra.split()

        # Configure step
        subprocess.check_call(['cmake', *generator, *cmake_args], cwd=build_temp)

        # Build step
        subprocess.check_call(['cmake', '--build', '.', '--config', cfg], cwd=build_temp)

        # On some platforms (esp. macOS), move the built .so where setuptools expects it
        for file in os.listdir(extdir):
            if file.startswith('libint_fuzzy') and (file.endswith('.so') or file.endswith('.pyd')):
                dest = os.path.join(self.get_ext_fullpath(ext.name))
                if not os.path.exists(dest):
                    shutil.move(os.path.join(extdir, file), dest)

setup(
    name='FuzzyQD-2.0',
    version='0.1.0',
    author='Ivan',
    description='Quantum dot modelling with fuzzy band structure and libint2 integrals',
    license='MIT',
    python_requires='>=3.9',
    packages=[],
    py_modules=['fuzzy2', 'parsers'],
    package_dir={'': 'src'},
    ext_modules=[CMakeExtension('libint_fuzzy', sourcedir='libint')],
    cmdclass={'build_ext': CMakeBuild},
    install_requires=[
        'numpy>=1.22',
        'scipy>=1.10',
        'matplotlib>=3.5',
        'pymatgen>=2024.0',
        'seekpath>=2.1',
        'pybind11>=2.12',
        # libint handled by conda env, not pip
    ],
    entry_points={
        'console_scripts': [
            'fuzzy2=fuzzy2:main',
        ],
    },
    zip_safe=False,
)

