# FuzzyQD 2.0

**FuzzyQD 2.0** is a hybrid Python/C++ toolkit for quantum dot (QD) band structure modeling, leveraging C++/pybind11 bindings to [libint2](https://github.com/evaleev/libint) for efficient Fourier transforms of atomic orbitals. Designed for high-throughput and HPC workflows, it features a robust command-line interface and parallel computation support.

---

## 🚀 Features

- Efficient "fuzzy" band structure generation for quantum dots using AO Fourier transforms.
- Full command-line interface for reproducibility and easy scripting.
- C++/pybind11 backend, linked to libint2, for high computational speed.
- HPC-friendly: works out of the box with Slurm and environment variables.
- All parameters are set via command-line keywords—no need to edit the script.

---

## ⚙️ Requirements

- **Python**: >= 3.9
- **Conda** (Miniconda or Anaconda, with conda-forge channel)
- **C/C++ compilers**: via the `compilers` metapackage (conda-forge)
- **CMake**: >= 3.22
- **libint**: >= 2.6 (conda-forge)
- **Boost**: (conda-forge)
- **pybind11**, **numpy**, **scipy**, **matplotlib**, **pymatgen**, **seekpath**

The recommended way is to use the provided `environment.yml`.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone git@github.com:nlesc-nano/FuzzyQD_2.0.git
cd FuzzyQD_2.0
```

### 2. Create the conda environment

```bash
mamba env create -f environment.yml
conda activate fuzzyqd2
```

*(If you don't have `mamba`, use `conda env create -f environment.yml` but `mamba` is faster.)*

### 3. Install the package with C++ compilation

```bash
pip install .
```
or for development (editable install):

```bash
pip install -e .
```

---

## 🧩 Usage

### Command Line Interface

All input parameters are **mandatory** and must be set via command line flags.  
Lattice vectors are specified as three arguments (`-A1`, `-A2`, `-A3`), each requiring three floats.

#### Arguments

| Flag           | Description                           | Example                        |
| -------------- | ------------------------------------- | ------------------------------ |
| `-A1`          | Lattice vector a1 (3 floats)          | `-A1 0.0 3.29 3.29`            |
| `-A2`          | Lattice vector a2 (3 floats)          | `-A2 3.29 0.0 3.29`            |
| `-A3`          | Lattice vector a3 (3 floats)          | `-A3 3.29 3.29 0.0`            |
| `-bulk_xyz`    | Path to bulk DFT xyz file             | `-bulk_xyz bulk.xyz`           |
| `-bulk_cif`    | Path to bulk CIF file                 | `-bulk_cif HgTe.cif`           |
| `-xyz`         | Path to QD xyz file                   | `-xyz geom.xyz`                |
| `-basis_txt`   | Path to basis set file (text)         | `-basis_txt BASIS_MOLOPT`      |
| `-basis_name`  | Basis set name                        | `-basis_name DZVP-MOLOPT-SR-GTH` |
| `-mo`          | Path to MO coefficients file          | `-mo MOs_cleaned.txt`          |
| `-ewin`        | Energy window for plotting (eV)       | `-ewin -9 -3`                  |
| `-nthreads`    | Number of threads (int, or see SLURM) | `-nthreads 8`                  |

#### Example command

```bash
fuzzy2   -A1 0.0 3.29 3.29   -A2 3.29 0.0 3.29   -A3 3.29 3.29 0.0   -bulk_xyz bulk.xyz   -bulk_cif HgTe.cif   -xyz geom.xyz   -basis_txt BASIS_MOLOPT   -basis_name DZVP-MOLOPT-SR-GTH   -mo MOs_cleaned.txt   -ewin -9 -3   -nthreads 8
```

---

### Parallel and HPC Usage

The script supports parallel execution by specifying `-nthreads`.  
For HPC/Slurm jobs, you can use the SLURM environment variable:

```bash
srun fuzzy2 ... -nthreads $SLURM_CPUS_PER_TASK
```

If `-nthreads` is omitted, the script will try to use `$SLURM_CPUS_PER_TASK` or all available CPUs.

#### Example Slurm script

```bash
#!/bin/bash
#SBATCH --job-name=fuzzyqd
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

source activate fuzzyqd2
srun fuzzy2 \
  -A1 0.0 3.29 3.29 \
  -A2 3.29 0.0 3.29 \
  -A3 3.29 3.29 0.0 \
  -bulk_xyz bulk.xyz \
  -bulk_cif HgTe.cif \
  -xyz geom.xyz \
  -basis_txt BASIS_MOLOPT \
  -basis_name DZVP-MOLOPT-SR-GTH \
  -mo MOs_cleaned.txt \
  -ewin -9 -3 \
  -nthreads $SLURM_CPUS_PER_TASK
```

---

## 📂 Input File Formats

- **XYZ files**: Standard xyz atom format. First line: number of atoms.
- **CIF files**: Standard Crystallographic Information File.
- **BASIS/MO**: Should match the format expected by your `parsers.py`.

---

## 📈 Output

- **Plots**: `fuzzy_band.png` is saved in the current directory.
- **Console log**: Full progress and key calculation steps are printed.
- **Additional files**: (k-points, tick labels) may be written if you uncomment those lines in the code.

---

## 🐍 Python API Usage

You can also use the code as a Python library:

```python
from fuzzy2 import main
from parsers import parse_basis, read_mos_txt2
import libint_fuzzy

# Use directly in your own workflow as needed.
```

---

## ⚡ Troubleshooting

- **`ModuleNotFoundError: libint_fuzzy`**  
  Check that the build succeeded, and that your Python environment is activated.

- **C++/CMake errors**  
  Make sure Boost, pybind11, and libint2 are installed via conda-forge.

- **Git conflicts**  
  If you have trouble pushing to GitHub, make sure you pulled the latest changes and resolved any merge conflicts.

---

## 🤝 Contributing & Support

- Open issues and pull requests on GitHub.
- For questions, email [Ivan Infante](mailto:i.infante@cicnano.es) or open an issue.
- For technical code support, you can also contact the [NLeSC Nano team](https://github.com/nlesc-nano).

---

## 📝 License

This project is licensed under the MIT License (see [LICENSE](LICENSE) file).

---

**Enjoy quantum dot band structures the easy (and fast) way!**
