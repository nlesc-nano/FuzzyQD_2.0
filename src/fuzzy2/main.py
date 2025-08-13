import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from time import perf_counter
from . import config, io_utils, structure_utils, plotting

# The C++ extension (built as libint_fuzzy.*.so)
try:
    import libint_fuzzy
except Exception as e:
    raise RuntimeError("libint_fuzzy extension not available. Build the C++ module first.") from e


def _count_ao_from_shells(shells):
    """
    Correctly infers the total AO dimension by summing the orbitals (2l+1)
    for each shell dictionary.
    """
    n_ao = 0
    for sh in shells:
        l = int(sh["l"])
        n_ao += (2 * l + 1)
    return int(n_ao)


def _dense(m):
    """Return a Fortran-ordered dense ndarray (no-op if already dense)."""
    from scipy.sparse import issparse
    if issparse(m):
        return m.toarray(order="F")
    return np.asarray(m, order="F")


def main():
    print("--- 🚀 FuzzyQD 2.0 (MO/SOC fuzzy bands) ---")

    # --- [0] Parse CLI Arguments ---
    args = config.get_args()
    user_lattice = np.array([args.A1, args.A2, args.A3], dtype=float)

    # --- [1] Load Reference Bulk Structure (from CIF, for info only) ---
    print("\n--- [1] Loading Reference Bulk Structure ---")
    struct_cif = io_utils.load_structure(args.bulk_cif)
    sga_cif = SpacegroupAnalyzer(struct_cif)
    struct_cif_conv = sga_cif.get_conventional_standard_structure()
    structure_utils.print_lattice_info("CIF conventional cell", struct_cif_conv.lattice.matrix)
    structure_utils.print_kpath_info(struct_cif_conv, "Symmetry from CIF")
    print(f"  ✓ Loaded reference: {struct_cif_conv.formula}")

    # --- [2] Load QD Geometry and Basis Set ---
    print("\n--- [2] Reading QD Structure (XYZ) and Basis Set ---")
    syms_qd, coords_qd_ang = io_utils.read_xyz(args.xyz)
    print(f"  ✓ Read {len(syms_qd)} atoms from '{args.xyz}'.")
    basis_dict = io_utils.parse_basis(args.basis_txt, args.basis_name)
    shells = structure_utils.build_shell_dicts(syms_qd, coords_qd_ang, basis_dict)
    n_ao_shells = _count_ao_from_shells(shells)
    print(f"  ✓ Parsed basis '{args.basis_name}' for {len(syms_qd)} atoms | Total AOs: {n_ao_shells}")
    # Spinor dimension (for SOC spinors we expect 2× spatial AO count)
    n_ao_spin = 2 * n_ao_shells

    # --- [3] Build the DFT Structure for K-Path Generation ---
    # This section now strictly follows the logic of your original script to ensure consistency.
    print("\n--- [3] Building DFT cell structure for k-path ---")
    syms_bulk_dft, coords_bulk_dft = io_utils.read_xyz(args.bulk_xyz)

    lat_dft = Lattice(user_lattice)
    struct_dft = Structure(
        lattice=lat_dft,
        species=syms_bulk_dft,
        coords=coords_bulk_dft,
        coords_are_cartesian=True,
    )

    # Standardize the DFT structure
    struct_dft_prim = SpacegroupAnalyzer(struct_dft).get_primitive_standard_structure()
    sga_dft = SpacegroupAnalyzer(struct_dft_prim)
    struct_dft_conv = sga_dft.get_conventional_standard_structure()
    print(f"  ✓ Built and standardized DFT structure with {len(struct_dft_conv)} atoms.")
    structure_utils.print_lattice_info("User-defined DFT cell", struct_dft_conv.lattice.matrix)

    # --- [4] Generate the K-Path ---
    print("\n--- [4] Generating k-path from DFT cell ---")
    kpts_cart, labels = structure_utils.get_kpoints_cart(
        struct_dft_conv, line_density=args.line_density
    )
    k_path_dist = structure_utils.k_path_distance(kpts_cart)
    print(f"  ✓ Generated {len(kpts_cart)} k-points on the high-symmetry path.")
    plotting.write_kpath_to_file(kpts_cart, labels, k_path_dist, outfile="kpath_debug.txt")

    # --- [5] AO Fourier Transform ---
    print("\n--- [5] Computing AO Fourier transforms ---")
    t0 = perf_counter() 
    kpts_bohr = np.asarray(kpts_cart, dtype=float) / config.BOHR_PER_ANGSTROM
    try:
        F = libint_fuzzy.ao_ft_complex(shells, np.asarray(kpts_bohr), args.nthreads)
        F = np.asarray(F)  # complex128
        print("  ✓ Using complex AO-FT (phase preserved)")
    except AttributeError:
        # Fallback to the real version if the complex one is not available
        F = libint_fuzzy.ao_ft(shells, np.asarray(kpts_bohr), args.nthreads)
        F = np.asarray(F, dtype=np.complex128)
        print(f"  ⚠︎ ao_ft_complex not found → falling back to real AO-FT")
    dt = perf_counter() - t0 
    print(f"  ✓ AO FT result F-matrix shape: {F.shape}. Done in {dt:.3f} seconds")

    # --- [6] Load States (MOs or SOC) and Project ---
    use_soc = args.soc_npz is not None
    if use_soc and getattr(args, 'mo', None):
        print("  ⚠︎ Both --mo and --soc_npz provided; ignoring --mo and using SOC spinors.")
        args.mo = None
    
    t0 = perf_counter() 
    if use_soc:
        print("\n--- [6] Loading SOC spinors ---")
        C_like, eps_eV, occ, _ = io_utils.load_soc_spinors_npz(args.soc_npz, verbose=True)
        # If spinor space is detected (SOC case)
        if C_like.shape[0] == n_ao_spin:
            print(f"  ✓ Detected SOC spinors (2×AO): {C_like.shape[0]} rows; duplicating AO-FT F to match spin blocks …")
            # Duplicate F for spin-up and spin-down
            F = np.vstack([F, F])
        
    else: # This is the original MO path
        print("\n--- [6] Loading molecular orbitals (MOs) ---")
        C, eps_Ha, occ = io_utils.read_mos_auto(args.mo, n_ao_shells, verbose=True)
        if C.shape[0] != n_ao_shells:
            raise ValueError(f"AO dimension mismatch: MOs {C.shape[0]} vs Basis {n_ao_shells}")
        eps_eV = eps_Ha * config.HARTREE_TO_EV
        C_like = _dense(C).astype(np.complex128, copy=False)
    dt = dt = perf_counter() - t0
    print(f"  ✓ States loaded: {C_like.shape[1]} | AO rows: {C_like.shape[0]} . Done in {dt:.3f} seconds")

    # --- Energy window filtering (apply to both MO and SOC) ---
    emin, emax = args.ewin
    # Default padding = 3*sigma to keep Gaussian broadening consistent
    pad = args.ewin_pad_ev if args.ewin_pad_ev is not None else (3.0 * args.sigma_ev if args.sigma_ev else 0.0)
    emin_eff = emin - pad
    emax_eff = emax + pad
    
    # energies_eV: shape (n_states,)
    # C_like:      shape (n_ao or 2*n_ao, n_states)
    mask = (eps_eV >= emin_eff) & (eps_eV <= emax_eff)
    n_before = eps_eV.shape[0]
    n_after  = int(mask.sum())
    
    if n_after == 0:
        print(f"  ⚠︎ ewin [{emin},{emax}] with pad {pad}eV kept 0 states; keeping all to avoid empty projection.")
    else:
        eps_eV = eps_eV[mask]
        if 'occupations' in locals() and occupations is not None:
            occupations = occupations[mask]
        if 'indices' in locals() and indices is not None:
            indices = indices[mask]  # keep mapping if you use it later
        C_like = C_like[:, mask]
        print(f"  ✓ Energy filter: kept {n_after}/{n_before} states in [{emin},{emax}] (+/−{pad} eV)")

    # --- [7] Project to k-space ---
    print("\n--- [7] Projecting to k-space: C.T @ F ---")
    t0 = perf_counter()
    Psi = C_like.conj().T @ F
    intensity = np.abs(Psi) ** 2
    dt = dt = perf_counter() - t0
    print(f"  ✓ Psi shape: {Psi.shape} | Intensity shape: {intensity.shape}. Done in {dt:.3f} seconds")

    # --- [8] Plotting ---
    print("\n--- [8] Plotting ---")
    outfile_name = "fuzzy_soc.png" if use_soc else "fuzzy_mo.png"
    plotting.plot_fuzzy_map_spinors(
        kpts_cart, labels, k_path_dist, eps_eV, intensity,
        ewin=args.ewin, sigma_ev=args.sigma_ev,
        gamma_norm=args.gamma_norm, scaled_vmin=args.scaled_vmin, outfile=outfile_name
    )

    # --- [9] Optional: HOMO/LUMO info ---
    if occ is not None:
        try:
            homo_idx = np.where(np.asarray(occ) > 0.0)[0][-1]
            print(f"\n  ℹ️  HOMO index: {homo_idx} | LUMO index: {homo_idx + 1}")
        except IndexError:
            print("\n  ℹ️  Could not determine HOMO/LUMO from occupations.")

    print("\n--- ✨ Done. ---")


if __name__ == "__main__":
    main()
