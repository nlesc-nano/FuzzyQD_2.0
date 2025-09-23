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

## 📖 Tutorial: Preparing MOs from CP2K

To generate the MO coefficients needed for **FuzzyQD 2.0**, follow these steps:

### 1. Run a single-point calculation in CP2K

After geometry optimization, perform a **single-point calculation** with the following SCF block:

```fortran
&SCF
  MAX_SCF 25
  EPS_SCF 1.0E-3
  ADDED_MOS 10000
  SCF_GUESS RESTART
#  &OT
#    MINIMIZER DIIS
#    N_DIIS 7
#    PRECONDITIONER FULL_SINGLE_INVERSE
#  &END OT
&END SCF
```

⚠️ **Important**:  
- Do **not** run OT calculations here.  
- `ADDED_MOS` must cover at least the number of LUMOs you want to include in the fuzzy band structure.

### 2. Print MOs to file

Add the following to your CP2K input:

```fortran
&PRINT
  &MO
    &EACH
      QS_SCF 100
    &END
    COEFFICIENTS
    MO_INDEX_RANGE 1085 6941
    NDIGITS 16
    ADD_LAST NUMERIC
    FILENAME MOs
  &END
&END PRINT
```

- `MO_INDEX_RANGE` should include **all occupied and unoccupied MOs** needed for the fuzzy band structure.  
- `ADDED_MOS` must be **≥ number of LUMOs** in `MO_INDEX_RANGE`.  

### 3. Clean the MOs file

After the run, clean the `MOs.txt` file to remove headers and redundant information.

For **RKS (closed-shell)**:

```bash
#!/bin/bash
INPUT="MOs.txt"
OUTPUT="MOs_cleaned.txt"

awk '
  BEGIN { skip=0; count=0 }
  /EIGENVALUES/ {
    count++
    if (count == 1) { skip=1; next }
    else if (count == 2) { skip=0; next }
  }
  skip == 0 { print }
' "$INPUT" | \
sed 's/MO|/ /g' | \
grep -v -E '^[[:space:]]*$' | \
grep -v 'E(Fermi)' | \
grep -v 'Band gap' > "$OUTPUT"

echo "✅ Cleaned file written to: $OUTPUT"
```

For **UKS (open-shell)**:

```bash
#!/bin/bash
set -euo pipefail

INPUT="${1:-MOs.txt}"
ALPHA_RAW="MOs_alpha_raw.txt"
BETA_RAW="MOs_beta_raw.txt"
ALPHA_OUT="MOs_alpha.txt"
BETA_OUT="MOs_beta.txt"

awk '
  BEGIN { section=0 }
  /^[[:space:]]*MO\|[[:space:]]*[Aa][Ll][Pp][Hh][Aa]/ { section=1; print > a; next }
  /^[[:space:]]*MO\|[[:space:]]*[Bb][Ee][Tt][Aa]/ { section=2; print > b; next }
  section==1 { print > a }
  section==2 { print > b }
' a="$ALPHA_RAW" b="$BETA_RAW" "$INPUT"

clean_mos() {
  local IN="$1"
  local OUT="$2"
  sed 's/MO|/ /g' "$IN" \
    | grep -viE "alpha|beta" \
    | grep -v -E "^[[:space:]]*$" \
    | grep -v "E(Fermi)" \
    | grep -v "Band gap" > "$OUT"
}

if [ -s "$ALPHA_RAW" ]; then
  clean_mos "$ALPHA_RAW" "$ALPHA_OUT"
else
  echo "Warning: no ALPHA section found in $INPUT" >&2
  : > "$ALPHA_OUT"
fi

if [ -s "$BETA_RAW" ]; then
  clean_mos "$BETA_RAW" "$BETA_OUT"
else
  echo "Note: no BETA section found in $INPUT" >&2
  : > "$BETA_OUT"
fi

rm -f "$ALPHA_RAW" "$BETA_RAW"

echo "Alpha MOs written to: $ALPHA_OUT"
echo "Beta  MOs written to: $BETA_OUT"
```

### 4. First-time use

The first time you run `fuzzy2` with an MO text file, reading may be slow (depending on number of basis functions and MOs).  
A cached `.npz` file will be created for **fast reuse** in subsequent runs.

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
| `--dos`        | Compute DOS                           | `--dos`                        |
| `--pdos_atoms` | Atoms for PDOS projection             | `--pdos_atoms all`             |
| `--coop`       | Compute COOP                          | `--coop all`                   |
| `-sigma_ev`    | PDOS broadening (eV)                  | `-sigma_ev 0.02`               |
| `-scaled_vmin` | Scaling factor for fuzzy intensity    | `-scaled_vmin 1e3`             |

#### Example command

```bash
fuzzy2 \
  -A1 6.58 0.0 0.0 \
  -A2 0.0 6.58 0.0 \
  -A3 0.0 0.0 6.58 \
  -bulk_cif HgTe.cif \
  -xyz geom.xyz \
  -basis_txt BASIS_MOLOPT \
  -basis_name DZVP-MOLOPT-SR-GTH \
  -mo MOs_cleaned_csr.npz \
  -ewin -10 -2 \
  -sigma_ev 0.02 \
  --dos \
  --pdos_atoms all \
  --coop all \
  -scaled_vmin 1e3
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
- **Additional files**: (k-points, tick labels, DOS/PDOS/COOP) depending on flags.  

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

- **Slow MO reading**  
  Use the `.npz` file generated after the first run for faster access.  

---

## 🤝 Contributing & Support

- Open issues and pull requests on GitHub.  
- For questions, email [Ivan Infante](mailto:i.infante@cicnano.es) or open an issue.  
- For technical code support, you can also contact the [NLeSC Nano team](https://github.com/nlesc-nano).  

---

## 📝 License

This project is licensed under the MIT License (see [LICENSE](LICENSE) file).
