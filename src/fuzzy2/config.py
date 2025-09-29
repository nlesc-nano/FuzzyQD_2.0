# src/fuzzy2/config.py

import argparse
import os
import numpy as np

# --- Physical Constants ---
HARTREE_TO_EV = 27.211386245988
BOHR_PER_ANGSTROM = 1.8897259886

# --- Default Parameters ---
DEFAULT_CPU_COUNT = os.cpu_count() or 1

def parse_lattice(arg_list):
    """Type checker for lattice vector arguments."""
    try:
        vec = [float(x) for x in arg_list]
        if len(vec) != 3:
            raise ValueError("Each lattice vector must have 3 components.")
        return vec
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid lattice vector: {e}")

def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FuzzyQD 2.0: Fuzzy Band Structure for Quantum Dots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Structure & Lattice ---
    parser.add_argument("-A1", nargs=3, type=float, required=True,
                        metavar=("X1","Y1","Z1"), help="Lattice vector A1")
    parser.add_argument("-A2", nargs=3, type=float, required=True,
                        metavar=("X2","Y2","Z2"), help="Lattice vector A2")
    parser.add_argument("-A3", nargs=3, type=float, required=True,
                        metavar=("X3","Y3","Z3"), help="Lattice vector A3")
     
    parser.add_argument("-bulk_xyz", type=str, default=None,
                   help="(Optional) Bulk structure as XYZ (Cartesian). "
                        "If omitted, bulk is built from --bulk_cif fractional coords and A1/A2/A3.")

    parser.add_argument("-bulk_cif", required=True, help="Path to the bulk structure in CIF format for symmetry analysis.")
    parser.add_argument("-xyz", required=True, help="Path to the quantum dot geometry in XYZ format.")

    # --- Basis Sets & MOs / SOC (mutually exclusive) ---
    parser.add_argument('-basis_txt', required=True, help='Path to the basis set definition file (e.g., BASIS_MOLOPT).')
    parser.add_argument('-basis_name', required=True, help='Name of the basis set to use (e.g., DZVP-MOLOPT-SR-GTH).')
    
    # Mutually exclusive group: either MO file or SOC spinor file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-mo', default=None, help='Path to the molecular orbital coefficients file (e.g., MOs_cleaned.txt).')
    group.add_argument('-soc_npz', default=None, help='Path to SOC spinors NPZ (indices, energies_Ha/eV, occupations, U). If provided, spinor pipeline is used.')
    
    # --- Calculation & Plotting ---
    parser.add_argument('--mo_ivk', nargs='*', type=int, default=None,
                        help='List of MO indices to plot intensity vs k. If omitted, plot first N HOMOs and N LUMOs.')
    parser.add_argument('--mo_ivk_n', type=int, default=5,
                        help='How many HOMOs and LUMOs to plot when --mo_ivk not given.')
    parser.add_argument("-ewin", nargs=2, type=float, required=True, metavar=("EMIN", "EMAX"),
                        help="Energy window [min, max] for plotting in eV.")
    parser.add_argument('-ewin_pad_ev', type=float, default=None,
                        help='Optional padding [eV] added on both sides of ewin (defaults to 3*sigma_ev if sigma_ev is set).')
    parser.add_argument("-nthreads", type=int, default=DEFAULT_CPU_COUNT,
                        help="Number of threads for parallel computation.")
    parser.add_argument("-line_density", type=int, default=600,
                        help="k-path line density (points per HS-path segment).")
    parser.add_argument("-sigma_ev", type=float, default=0.10,
                        help="Energy broadening (Gaussian σ) for fuzzy map in eV.")
    parser.add_argument("-gamma_norm", type=float, default=None,
                        help="Optional gamma for intensity normalization: I -> I^gamma.")
    parser.add_argument("-scaled_vmin", type=float, default=None,
                        help="Optional scaled vmin for intensity plot.")

    # --- Analysis ---
    parser.add_argument("--dos", action="store_true", help="Enable DOS and PDOS calculation.")
    parser.add_argument("--pdos_atoms", nargs='+', type=str, help="List of atom symbols for PDOS (e.g., Hg Te), or 'all'.")
    parser.add_argument("--coop", nargs='+', type=str, help="List of atom pairs for COOP (e.g., Hg-Te), or 'all'.")
    parser.add_argument("--population_analysis", type=str, default="mulliken", choices=["mulliken", "lowdin"],
                        help="Population analysis method for PDOS and COOP.")

    # Threading / BLAS control
    parser.add_argument("--no_thread_autoset", action="store_true",
        help="Do not auto-configure OpenMP/BLAS threads; use the environment as-is.")
    parser.add_argument("--blas_threads", type=int, default=1,
        help="Threads for MKL/OpenBLAS (default: 1; set 0 to leave unchanged).")

    # --- Cube export ---
    parser.add_argument("--cube", action="store_true",
        help="Export selected MOs as Gaussian cube files after plotting.")
    parser.add_argument("--cube_spacing", type=float, default=0.40,
        help="Cube grid spacing in bohr (default 0.40 ≈ 0.21 Å).")
    parser.add_argument("--cube_padding", type=float, default=6.0,
        help="Cube margin around molecule in bohr (default 6.0).")
    parser.add_argument("--cube_part", choices=["real","imag","abs","abs2"], default="real",
        help="If MO coefficients are complex, which part to write (default real).")
    
    args = parser.parse_args()
    return args
