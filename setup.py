import os
import pathlib
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

HERE = pathlib.Path(__file__).parent.resolve()
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else "FuzzyQD 2"

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = str(pathlib.Path(sourcedir).resolve())

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
#        super().run()

    def build_cmake(self, ext: CMakeExtension):
        # NEW: Get the full path to the expected build artifact
        ext_path = self.get_ext_fullpath(ext.name)
    
        # NEW: Check if the file already exists to skip recompilation
        if os.path.exists(ext_path):
            print(f"Skipping build: {ext_path} already exists.")
            return
    
        cfg = "Debug" if self.debug else "Release"
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
    
        extdir = pathlib.Path(ext_path).parent.resolve() # Modified this line to use the path variable
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
    
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            cmake_args.append(f"-DCMAKE_PREFIX_PATH={conda_prefix}")
    
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."], cwd=build_temp)
    
setup(
    name="FuzzyQD-2",
    version="2.0.1",
    description="Fuzzy band structures for QDs (MO & SOC spinors)",
    long_description=README,
    long_description_content_type="text/markdown",
    author="",
    license="BSD-3-Clause",

    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,

    ext_modules=[CMakeExtension("libint_fuzzy", sourcedir="libint")],
    cmdclass={"build_ext": CMakeBuild},

    install_requires=[
        "numpy>=1.22",
        "scipy>=1.10",
        "matplotlib>=3.5",
        "pymatgen>=2024.0",
        "seekpath>=2.1",
        "pybind11>=2.12",
    ],
    python_requires=">=3.9",

    entry_points={
        "console_scripts": [
            "fuzzy2=fuzzy2.main:main",
        ]
    },
)
