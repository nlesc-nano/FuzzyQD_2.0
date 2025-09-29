import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from time import perf_counter
from itertools import combinations
from . import config, io_utils, structure_utils, plotting, analysis

# The C++ extension (built as libint_fuzzy.*.so)
try:
    import libint_fuzzy
except Exception as e:
    raise RuntimeError("libint_fuzzy extension not available. Build the C++ module first.") from e


def _count_ao_from_shells(shells):
    """
    Correctly infers total AO dimension by summing (2l+1) for each shell.
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
    if not args.no_thread_autoset:
        io_utils.configure_threading(
            nthreads=args.nthreads if hasattr(args, "nthreads") else 1,
            blas_threads=args.blas_threads,
            quiet=False
        )

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
    unique_atom_types = sorted(list(set(syms_qd)))
    print(f"  ✓ Read {len(syms_qd)} atoms from '{args.xyz}'. Unique types: {unique_atom_types}")
    basis_dict = io_utils.parse_basis(args.basis_txt, args.basis_name)
    shells = structure_utils.build_shell_dicts(syms_qd, coords_qd_ang, basis_dict)
    n_ao_shells = _count_ao_from_shells(shells)
    print(f"  ✓ Parsed basis '{args.basis_name}' for {len(syms_qd)} atoms | Total AOs: {n_ao_shells}")
    # Spinor dimension (for SOC spinors we expect 2× spatial AO count)
    n_ao_spin = 2 * n_ao_shells

    # --- [3] Build the DFT Structure for K-Path Generation ---
    print("\n--- [3] Building DFT cell structure for k-path ---")
    lat_dft = Lattice(user_lattice)  # rows are A1,A2,A3 (Å)

    if getattr(args, "bulk_xyz", None):
        # Backward-compatible path: user-supplied Cartesian XYZ
        syms_bulk_dft, coords_bulk_dft = io_utils.read_xyz(args.bulk_xyz)
        struct_dft = Structure(
            lattice=lat_dft,
            species=syms_bulk_dft,
            coords=coords_bulk_dft,
            coords_are_cartesian=True,
        )
        print("  ✓ Using bulk XYZ provided on CLI.")
    else:
        # New default: build from CIF fractional coords + user lattice
        # We already standardized the CIF → use its conventional cell fractional sites
        struct_ref = struct_cif_conv
        struct_dft = Structure(
            lattice=lat_dft,
            species=list(struct_ref.species),
            coords=struct_ref.frac_coords,
            coords_are_cartesian=False,
        )
        # Keep atoms inside cell to avoid tiny negative fracs carrying over
        struct_dft = Structure.from_sites([s.to_unit_cell() for s in struct_dft.sites])
        print("  ✓ Built bulk from CIF fractional coords and A1/A2/A3.")

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

    else:  # This is the original MO path
        print("\n--- [6] Loading molecular orbitals (MOs) ---")
        C, eps_Ha, occ = io_utils.read_mos_auto(args.mo, n_ao_shells, verbose=True)
        if C.shape[0] != n_ao_shells:
            raise ValueError(f"AO dimension mismatch: MOs {C.shape[0]} vs Basis {n_ao_shells}")
        eps_eV = eps_Ha * config.HARTREE_TO_EV
        C_like = _dense(C).astype(np.complex128, copy=False)
    dt = perf_counter() - t0
    print(f"  ✓ States loaded: {C_like.shape[1]} | AO rows: {C_like.shape[0]} . Done in {dt:.3f} seconds")

    # --- Energy window filtering (apply to both MO and SOC) ---
    emin, emax = args.ewin
    pad = args.ewin_pad_ev if args.ewin_pad_ev is not None else (3.0*args.sigma_ev if args.sigma_ev else 0.0)
    emin_eff, emax_eff = emin - pad, emax + pad

    def _homo_lumo_from_occ(eps_eV_arr, occ_arr, tol=1e-8):
        """Return (homo_i, lumo_i) indices in the given arrays or (None, None)."""
        if occ_arr is None or np.size(occ_arr) == 0:
            return None, None
        filled = np.where(np.asarray(occ_arr) > tol)[0]
        if filled.size == 0:
            return None, None
        h = filled[-1]
        l = h + 1 if h + 1 < eps_eV_arr.size else None
        if l is None:
            return None, None
        return h, l

    # 1) HOMO/LUMO on the FULL state list (pre-filter)
    homo_i_full, lumo_i_full = _homo_lumo_from_occ(eps_eV, occ)
    if homo_i_full is not None and lumo_i_full is not None:
        EH0, EL0 = eps_eV[homo_i_full], eps_eV[lumo_i_full]
        midgap_full = 0.5*(EH0 + EL0)
        print(f"[States|FULL] HOMO idx={homo_i_full}, E={EH0:.3f} eV | "
              f"LUMO idx={lumo_i_full}, E={EL0:.3f} eV | gap={EL0-EH0:.3f} eV")
    else:
        midgap_full = 0.5*(emin + emax)
        print(f"[States|FULL] Occupations unavailable → using ewin center: {midgap_full:.3f} eV")

    # 2) Build mask and index map FULL→FILTERED
    mask = (eps_eV >= emin_eff) & (eps_eV <= emax_eff)
    idx_map = np.flatnonzero(mask)           # positions kept
    n_before, n_after = eps_eV.size, idx_map.size

    # 3) Apply filter (only if anything survives)
    if n_after == 0:
        print(f"  ⚠︎ ewin [{emin},{emax}] ±{pad} eV kept 0 states; keeping all to avoid empty projection.")
        eps_eV_f = eps_eV
        occ_f    = occ
        C_like_f = C_like
        # HOMO/LUMO in filtered == full in this edge case
        homo_i_f, lumo_i_f = homo_i_full, lumo_i_full
        midgap = midgap_full
    else:
        eps_eV_f = eps_eV[idx_map]
        occ_f    = occ[idx_map] if occ is not None else None
        C_like_f = C_like[:, idx_map]
        print(f"  ✓ Energy filter: kept {n_after}/{n_before} states in [{emin},{emax}] (±{pad} eV)")

        # 4) HOMO/LUMO on the FILTERED list (for plotting/selection)
        homo_i_f, lumo_i_f = _homo_lumo_from_occ(eps_eV_f, occ_f)
        midgap = (0.5*(eps_eV_f[homo_i_f] + eps_eV_f[lumo_i_f])
                  if (homo_i_f is not None and lumo_i_f is not None) else midgap_full)

    # From here on, use eps_eV_f / occ_f / C_like_f
    eps_eV = eps_eV_f
    occ    = occ_f
    C_like = C_like_f

    # --- [7] Project to k-space ---
    print("\n--- [7] Projecting to k-space: C.T @ F ---")
    t0 = perf_counter()
    Psi = C_like.conj().T @ F
    intensity = np.abs(Psi) ** 2
    dt = perf_counter() - t0
    print(f"  ✓ Psi shape: {Psi.shape} | Intensity shape: {intensity.shape}. Done in {dt:.3f} seconds")

    # --- [8] Plotting ---
    print("\n--- [8] Plotting ---")
    outfile_name = "fuzzy_soc.png" if use_soc else "fuzzy_mo.png"
    plotting.plot_fuzzy_map_spinors(
        kpts_cart, labels, k_path_dist, eps_eV, intensity,
        ewin=args.ewin, sigma_ev=args.sigma_ev,
        gamma_norm=args.gamma_norm, scaled_vmin=args.scaled_vmin,
        outfile=outfile_name, midgap=midgap
    )

    # --- [9] Analysis (DOS, PDOS, COOP) ---
    if args.dos or args.coop:
        print("\n--- [9] Analysis ---")
        t0 = perf_counter()
        S = libint_fuzzy.overlap(shells, args.nthreads)
        dt = perf_counter() - t0
        print(f"  ✓ Overlap matrix computed in {dt:.3f} seconds.")

        if args.dos:
            if not args.pdos_atoms:
                raise ValueError("--dos requires --pdos_atoms to be specified (e.g., --pdos_atoms Hg Te or --pdos_atoms all).")

            pdos_atom_list = unique_atom_types if args.pdos_atoms == ["all"] else args.pdos_atoms
            analysis.plot_dos_and_pdos(
                eps_eV, occ, C_like, S, shells,
                pdos_atom_list, args.ewin,
                method=args.population_analysis,
                sigma=0.08 
            )
            pdos_weights = analysis.compute_pdos_weights(C_like, S, method=args.population_analysis)
            analysis.print_pdos_population_analysis(pdos_weights, shells, eps_eV, occ)

        if args.coop:
            if args.coop == ["all"]:
                coop_pair_list = [f"{a}-{b}" for a, b in combinations(unique_atom_types, 2)]
            else:
                coop_pair_list = args.coop
            coop_weights = analysis.compute_coop(C_like, S, shells, coop_pair_list)
            analysis.plot_coop(
                eps_eV, C_like, S, shells,
                coop_pair_list, args.ewin,
                method=args.population_analysis,
                sigma=(args.sigma_ev or 0.1)
            )
            analysis.print_coop_analysis(coop_weights, eps_eV, occ)

        # --- [9.5] Combined fuzzy + PDOS figure (shared energy axis) ---
        try:
            # --- [8b] Combined panels on demand ---
            if args.dos and args.coop:
                # 3-panel: fuzzy + PDOS + COOP (shared y-axis, legends + colorbar on far right)
                coop_pair_list = ([f"{a}-{b}" for a in sorted(set(syms_qd)) for b in sorted(set(syms_qd)) if a != b]
                                  if args.coop == ["all"] else args.coop)
                pdos_atom_list = (sorted(set(syms_qd)) if args.pdos_atoms == ["all"] else args.pdos_atoms)
            
                analysis.plot_fuzzy_pdos_coop_combo(
                    kpts_cart, labels, k_path_dist,
                    eps_eV, intensity,
                    C_like, S, shells,
                    pdos_atom_list=pdos_atom_list,
                    coop_pair_list=coop_pair_list,
                    ewin=args.ewin,
                    sigma_ev=args.sigma_ev,
                    sigma_pdos=0.08,      # keep PDOS σ independent of fuzzy
                    midgap=midgap,
                    occ=occ,
                    outfile="fuzzy_pdos_coop.png",
                    scaled_vmin=args.scaled_vmin,
                )
            elif args.dos:
                # 2-panel: fuzzy + PDOS (backward compatible)
                pdos_atom_list = (sorted(set(syms_qd)) if args.pdos_atoms == ["all"] else args.pdos_atoms)
                analysis.plot_fuzzy_and_pdos_combo(
                    kpts_cart, labels, k_path_dist,
                    eps_eV, intensity,
                    C_like, S, shells,
                    pdos_atom_list=pdos_atom_list,
                    ewin=args.ewin,
                    sigma_ev=args.sigma_ev,
                    sigma_pdos=0.08,
                    midgap=midgap,
                    occ=occ,
                    outfile="fuzzy_plus_pdos.png",
                    scaled_vmin=args.scaled_vmin,
                )
            # else: no combined side-panels requested
            
        except Exception as e:
            print(f"  ⚠︎ Combined fuzzy+PDOS plot failed: {e}")

    # --- [10] Individual MO Intensity Plots ---
    if args.mo_ivk or (homo_i_f is not None and not use_soc):
        if args.mo_ivk:
            mo_list = args.mo_ivk
        else:
            N = args.mo_ivk_n
            homos = list(range(max(0, homo_i_f - N + 1), homo_i_f + 1))
            lumos = list(range(homo_i_f + 1, min(homo_i_f + 1 + N, len(eps_eV))))
            mo_list = homos + lumos

        if mo_list:
            print("\n--- [10] Plotting individual MO intensities ---")
            I_nk = intensity
            eps_Ha_out = eps_eV / config.HARTREE_TO_EV
            plotting.plot_mo_intensity(
                kpts_cart, labels, k_path_dist,
                I_nk, eps_Ha_out, mo_list,
                gamma_norm=args.gamma_norm,
                outfile_prefix="mo_intensity"
            )
            # --- Cube export (after plotting) ---
            if args.cube:
                from .cube import write_mo_cubes
                prefix = getattr(args, "prefix", None) or "system"
                paths = write_mo_cubes(
                    prefix=prefix,
                    syms_qd=syms_qd,
                    coords_qd_ang=coords_qd_ang,
                    shells=shells,
                    C_ao_mo=C_like,
                    mo_indices=mo_list,
                    spacing_bohr=args.cube_spacing,
                    padding_bohr=args.cube_padding,
                    part=args.cube_part,
                    nthreads=args.nthreads if hasattr(args, "nthreads") else 1,
                )
                print(f"  ✓ Wrote {len(paths)} cube(s) → ./cubes")

    # --- [11] Optional: HOMO/LUMO info ---
    if occ is not None:
        try:
            homo_idx = np.where(np.asarray(occ) > 0.0)[0][-1]
            print(f"\n  ℹ️  HOMO index: {homo_idx} | LUMO index: {homo_idx + 1}")
        except IndexError:
            print("\n  ℹ️  Could not determine HOMO/LUMO from occupations.")

    print("\n--- ✨ Done. ---")


if __name__ == "__main__":
    main()

