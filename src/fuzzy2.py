from pathlib import Path
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifParser
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter, maximum_filter
from parsers import parse_basis, read_mos_txt, read_mos_txt2 
import collections, argparse 
 
import numpy as np
import inspect, os, collections 
import libint_fuzzy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.fft import fftn, fftshift, ifftshift

LATTICE = np.array([
    [0.000, 3.290, 3.290],
    [3.290, 0.000, 3.290],
    [3.290, 3.290, 0.000]
])  # Å  ← put your own 3×3 matrix here
BULK_DFT_XYZ_PATH = "bulk.xyz"
BULK_CIF = "HgTe.cif"
XYZ_PATH = "geom.xyz"
BASIS_TXT = "BASIS_MOLOPT"
BASIS_NAME = "DZVP-MOLOPT-SR-GTH"
MO_PATH = "MOs_cleaned.txt" 
E_WINDOW = (-9, -3)  
_NTHREADS = os.cpu_count() or 1
_H2EV = 27.211386245988
_BOHR_PER_ANG = 1.8897259886

# --- 0. Structure I/O -----------------------------------------------------
def load_structure(cif_path=None, mpid=None, api_key=None):
    if cif_path:
        return CifParser(cif_path).parse_structures(primitive=True)[0]
    if mpid:
        with MPRester(api_key) as mpr:
            return mpr.get_structure_by_material_id(mpid)
    raise ValueError("Provide either cif_path or mpid")

def parse_lattice(args):
    # expects args.A1, args.A2, args.A3 as lists of floats
    try:
        A1 = [float(x) for x in args.A1]
        A2 = [float(x) for x in args.A2]
        A3 = [float(x) for x in args.A3]
        L = np.array([A1, A2, A3])
        if L.shape != (3,3):
            raise ValueError("Each lattice vector must have 3 components.")
        return L
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid lattice vectors: {e}")

def get_args():
    parser = argparse.ArgumentParser(description="FuzzyQD 2.0: Fuzzy Band Structure for Quantum Dots")
    parser.add_argument("-A1", nargs=3, required=True, metavar=("X1", "Y1", "Z1"),
                        help="Lattice vector a1 (3 floats)")
    parser.add_argument("-A2", nargs=3, required=True, metavar=("X2", "Y2", "Z2"),
                        help="Lattice vector a2 (3 floats)")
    parser.add_argument("-A3", nargs=3, required=True, metavar=("X3", "Y3", "Z3"),
                        help="Lattice vector a3 (3 floats)")
    parser.add_argument("-bulk_xyz", required=True, help="Path to bulk DFT xyz file")
    parser.add_argument("-bulk_cif", required=True, help="Path to bulk CIF file")
    parser.add_argument("-xyz", required=True, help="Path to QD xyz file")
    parser.add_argument("-basis_txt", required=True, help="Path to basis text file")
    parser.add_argument("-basis_name", required=True, help="Name of basis set")
    parser.add_argument("-mo", required=True, help="Path to MO coefficients file")
    parser.add_argument("-ewin", nargs=2, type=float, required=True, metavar=("EMIN", "EMAX"),
                        help="Energy window for plotting (eV)")
    parser.add_argument("-nthreads", type=int, required=True, help="Number of threads")
    args = parser.parse_args()
    return args

def print_lattice_info(name, lattice):
    print(f"[INFO] {name} lattice vectors (Å):")
    for i, v in enumerate(lattice):
        print(f"    a{i+1}: [{v[0]:8.4f}  {v[1]:8.4f}  {v[2]:8.4f}]")
    rec = 2 * np.pi * np.linalg.inv(lattice).T
    print(f"[INFO] {name} reciprocal lattice vectors (Å⁻¹):")
    for i, v in enumerate(rec):
        print(f"    b{i+1}: [{v[0]:8.4f}  {v[1]:8.4f}  {v[2]:8.4f}]")
    print()

# --- 1. Symmetrisation ----------------------------------------------------
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def to_primitive(struct):
    return SpacegroupAnalyzer(struct).get_primitive_standard_structure()

# --- 2A. k-path via pymatgen (robust wrapper) -----------------------------
def get_kpoints_cart(struct, line_density=50, recipe="setyawan_curtarolo"):
    from pymatgen.symmetry.bandstructure import HighSymmKpath
    kp = HighSymmKpath(struct, path_type=recipe)
    try:
        params = kp.get_kpoints.__code__.co_varnames
        if "coords_are_cartesian" in params:
            # pymatgen >= 2024.4
            kpts, labels = kp.get_kpoints(line_density=line_density, coords_are_cartesian=True)
            return np.asarray(kpts), labels
        elif "cart_coords" in params:
            # pymatgen <= 2024.3
            kpts, labels = kp.get_kpoints(line_density=line_density, cart_coords=True)
            return np.asarray(kpts), labels
    except Exception:
        pass
    # fallback: get fractional, then convert
    frac_kpts, labels = kp.get_kpoints(line_density=line_density)
    B = struct.lattice.reciprocal_lattice.matrix
    kpts = np.dot(frac_kpts, B)
    return np.asarray(kpts), labels

# --- 2B. k-path with SeeK-path --------------------------------------------
def kpath_seek(struct, ref_dist=0.025):
    import seekpath
    spg_tuple = (struct.lattice.matrix,
                 struct.frac_coords,
                 struct.atomic_numbers)
    out = seekpath.get_explicit_k_path(spg_tuple, reference_distance=ref_dist)
    return (np.array(out["explicit_kpoints_abs"]),
            out["explicit_kpoints_labels"])

# ----- Parse XYZ Structure -----
def read_xyz(path):
    with open(path) as f:
        lines=f.readlines()
    nat=int(lines[0].split()[0])
    syms=[]; coords=[]
    for l in lines[2:2+nat]:
        p=l.split()
        syms.append(p[0])
        coords.append(tuple(map(float,p[1:4])))
    return syms, np.asarray(coords)

#---- Create Shell Dictionary -------
def build_shell_dicts(syms, coords_ang, basis_dict):
    """
    Corrected version that adds 'sym' and 'atom_idx' to each shell dict,
    which is required for the analysis functions.
    """
    shells = []
    for atom_idx, (sym, xyz_ang) in enumerate(zip(syms, coords_ang)):
        if sym not in basis_dict: raise KeyError(f"no basis for {sym}")
        xyz_bohr = np.asarray(xyz_ang, dtype=float) * _BOHR_PER_ANG
        for l, exps, coefs in basis_dict[sym]:
            shells.append(dict(sym=sym, atom_idx=atom_idx, l=l, exps=exps, coefs=coefs, center=xyz_bohr))
    return shells

def make_ao_info(shell_dicts_spherical):
    # Returns [{'sym':..., 'atom_idx':..., 'l':..., 'm':...}, ...]
    return [
        {'sym': sh['sym'], 'atom_idx': sh['atom_idx'], 'l': sh['l'], 'm': m}
        for sh in shell_dicts_spherical
        for m in range(-sh['l'], sh['l']+1)
    ]

# ----- Plot helper ------
def dedup_kpath(kpts, labels, tol=1e-8):
    """Remove consecutive duplicate k‑points (norm < tol)."""
    keep_idx = [0]
    for i in range(1, len(kpts)):
        if np.linalg.norm(kpts[i] - kpts[i-1]) > tol:
            keep_idx.append(i)
    return kpts[keep_idx], [labels[i] for i in keep_idx]

def _fade_cmap():
    base = plt.colormaps.get_cmap("inferno")
    cols = [(0,0,0)] + list(base(np.linspace(0,1,256)))  # No white at top!
    pos  = [0.0] + list(np.linspace(0.0,1.0,256))
    return mcolors.LinearSegmentedColormap.from_list("inf_bw", list(zip(pos,cols)))

def plot_mo_intensity(kpts_cart, labels, kpath, intensity,
                      eps_Ha, mo_indices,
                      gamma_norm=None):
    """
    Plot |Ψ_n(k)|² along the k‑path for each MO index in `mo_indices`.
    Parameters
    ----------
    kpts_cart : (N,3) array  – raw cartesian k‑points (already deduped)
    labels    : list[str]    – one label per k‑point
    kpath     : (N,) array   – cumulative distance (same grid as intensity)
    intensity : (N_MO,N)     – |Ψ(k)|² per MO
    eps_Ha    : (N_MO,)      – orbital energies (Hartree)
    mo_indices: list[int]    – 0‑based indices (e.g. [HOMO] or [HOMO,LUMO])
    gamma_norm: float|None   – if set, apply y‑scale I^gamma to flatten tails
    """
    for n in mo_indices:
        I_k = intensity[n]                    # (N,) array
        if gamma_norm is not None:
            I_k = I_k**gamma_norm
        E_eV = eps_Ha[n] * _H2EV
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(kpath, I_k, lw=0.8)
        ax.set_title(f"MO {n+1}   E = {E_eV:6.3f} eV")
        ax.set_ylabel(r"$|\Psi(k)|^{2}$")
        ax.set_xlabel("k‑path")
        # put Γ, X, … ticks as we did for fuzzy plot
        hs_pos, hs_lbl, prev = [], [], None
        for x,lbl in zip(kpath, labels):
            if lbl and lbl!=prev:
                hs_pos.append(x); hs_lbl.append(lbl); prev=lbl
        ax.xaxis.set_major_locator(MaxNLocator(nbins=len(hs_pos)+1))
        ax.set_xticks(hs_pos); ax.set_xticklabels(hs_lbl)
        for xc in hs_pos:
            ax.axvline(xc, color='gray', lw=0.3, alpha=0.5)
        plt.tight_layout()
        plt.show()


def fuzzy_band_imshow(kpts_cart, labels, eps_Ha, occ, intensity,
                      dE=0.05, energy_window=None, vmin_scale=1e3,
                      blur_sigma=0.4, outfile=None,
                      kpts_file="kpoints.txt", ticks_file="ticks.txt"):
    # 0) de‑duplicate end‑points
    kpts_cart, labels, keep = dedup_kpath_strict(np.asarray(kpts_cart), labels)
    intensity = intensity[:, keep]

    # 1) uniform pixel grid parameters
    ncol = kpts_cart.shape[0]
    x_min, x_max = 0.0, float(ncol)        # arbitrary uniform grid
    dx = 1.0                               # pixel width = 1
    pixel_centres = x_min + (np.arange(ncol)+0.5)*dx

    # 2) write k‑points with total intensity + tick label
    I_tot = intensity.sum(axis=0)
#    with open(kpts_file,"w") as f:
#        f.write("# idx   kx(Å⁻¹)        ky          kz         "
#                "pixel_x      I_tot        tick\n")
#        for i,(k,px,I,lbl) in enumerate(zip(kpts_cart,pixel_centres,I_tot,labels),1):
#            mark = lbl if lbl else "-"
#            f.write(f"{i:4d}  {k[0]:12.8f}  {k[1]:12.8f}  {k[2]:12.8f}  "
#                    f"{px:10.2f}  {I:12.5e}   {mark}\n")
#    print("k‑points →", kpts_file)

    # 3) energy binning
    E_eV = eps_Ha*_H2EV
    mask = slice(None) if energy_window is None else (
           (E_eV>=energy_window[0])&(E_eV<=energy_window[1]))
    E_use, I_use = E_eV[mask], intensity[mask,:]
    edges = np.arange(E_use.min(), E_use.max()+dE, dE)
    centres = 0.5*(edges[:-1]+edges[1:])
#    Z=np.zeros((len(centres),ncol))
#    idx=np.digitize(E_use,edges)-1
#    for i in range(len(centres)): Z[i]=I_use[idx==i].sum(axis=0)
#    if blur_sigma: Z=gaussian_filter(Z,sigma=blur_sigma)


    sigma = 0.01   # eV broadening
    Z = np.zeros((len(centres), ncol))
    for E_n, psi_k in zip(E_use, I_use):      # psi_k shape (N_k,)
        w = np.exp(-0.5*((centres - E_n)/sigma)**2)
        Z += np.outer(w, psi_k)

    # 4) colour scale
    vmax = np.percentile(Z, 99.9)
    vmin = max(np.percentile(Z[Z > 0], 5), vmax / 1e4)
    midpoint = np.percentile(Z, 65)
#    norm = GammaNorm(gamma=0.3, vmin=vmin, vmax=vmax)
    norm=mcolors.LogNorm(vmin=vmin,vmax=vmax)
   
    # 5) tick positions on pixel grid
    tick_pos, tick_lab, prev = [], [], None
    for i,(lbl) in enumerate(labels):
        if lbl and lbl!=prev:
            tick_pos.append(pixel_centres[i])
            tick_lab.append(fr"$\mathbf{{{lbl}}}$")
            prev=lbl
#    with open(ticks_file,"w") as f:
#        f.write("pixel_x  label\n")
#        for x,l in zip(tick_pos,tick_lab): f.write(f"{x:10.2f}  {l}\n")
#    print("ticks   →", ticks_file)

    # 6) plot
    fig,ax=plt.subplots(figsize=(9,5),facecolor="white")
    extent=[x_min,x_max,centres.min(),centres.max()]
    im=ax.imshow(Z,origin='lower',aspect='auto',extent=extent,
                 cmap=_fade_cmap(),norm=norm)
    if energy_window: ax.set_ylim(energy_window)

    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab,fontsize=11)
    for xc in tick_pos: ax.axvline(xc,color='w',lw=0.3,alpha=0.5)
    ax.set_xlabel("k‑path (uniform index)"); ax.set_ylabel("Energy (eV)")
    ax.set_facecolor('black')
    cb=plt.colorbar(im,ax=ax,pad=0.02,extend='both')
    cb.set_label(r"$|\Psi(k)|^2$ (log)",fontsize=11)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile,dpi=300); print("plot →",outfile)
    else:
        plt.show()

def dedup_kpath_strict(kpts, labels, tol=1e-8):
    """
    Deduplicate only if both k-point and label are the same.
    Ensures high-symmetry points at segment boundaries are kept.
    Returns: kpts_new, labels_new, kept_indices
    """
    keep = [0]
    for i in range(1, len(kpts)):
        same_coord = np.linalg.norm(kpts[i] - kpts[keep[-1]]) < tol
        same_label = labels[i] == labels[keep[-1]]
        if not (same_coord and same_label):
            keep.append(i)
    return kpts[keep], [labels[i] for i in keep], np.array(keep)


def k_path_distance(kpts_cart):
    dk = np.linalg.norm(np.diff(kpts_cart, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(dk)))

def print_kpath_info(struct, tag):

    # Crystal symmetry info
    sga = SpacegroupAnalyzer(struct, symprec=1e-3, angle_tolerance=5)
    spg_symbol, spg_number = sga.get_space_group_symbol(), sga.get_space_group_number()
    crystal_system = sga.get_crystal_system()
    point_group = sga.get_point_group_symbol()

    print(f"\n[{tag}] Crystal Symmetry Info")
    print(f"  Space group:    {spg_symbol} (No. {spg_number})")
    print(f"  Crystal system: {crystal_system}")
    print(f"  Point group:    {point_group}")
    print(f"  Structure formula: {struct.composition.reduced_formula}")
    print(f"  Lattice parameters (Å): a={struct.lattice.a:.4f}, b={struct.lattice.b:.4f}, c={struct.lattice.c:.4f}, alpha={struct.lattice.alpha:.2f}, beta={struct.lattice.beta:.2f}, gamma={struct.lattice.gamma:.2f}")

    # k-path
    kp = HighSymmKpath(struct)
    if kp.kpath is None:
        print("  No symmetry k-path found for this structure.")
        return
    kdict = kp.kpath["kpoints"]
    B = struct.lattice.reciprocal_lattice.matrix

    print(f"\n[{tag}] High-symmetry points")
    print("  Label    frac_k1   frac_k2   frac_k3     kx(Å⁻¹)     ky(Å⁻¹)     kz(Å⁻¹)")
    print("  -----   --------  --------  --------   ----------  ----------  ----------")
    for lbl, frac in kdict.items():
        cart = np.dot(frac, B)
        print(f"  {lbl:<6}  {frac[0]:8.3f}  {frac[1]:8.3f}  {frac[2]:8.3f} "
              f"  {cart[0]:10.4f}  {cart[1]:10.4f}  {cart[2]:10.4f}")

    # Print the k-path segments
    print(f"\n[{tag}] K-path segments:")
    for seg in kp.kpath['path']:
        print("   " + " → ".join(seg))

    # Optionally: print out the actual sequence of k-points (if desired)
    # kpts, labels = kp.get_kpoints(line_density=20)
    # print(f"\n[{tag}] K-path detailed sequence (label, frac_k):")
    # for i, (k, l) in enumerate(zip(kpts, labels)):
    #     if l:
    #         print(f"  [{i:4d}] {l:4s} {k}")

def main():
    # --- [0] Read from CLI ---- 
    args = get_args()
    LATTICE = parse_lattice(args)
    BULK_DFT_XYZ_PATH = args.bulk_xyz
    BULK_CIF = args.bulk_cif
    XYZ_PATH = args.xyz
    BASIS_TXT = args.basis_txt
    BASIS_NAME = args.basis_name
    MO_PATH = args.mo
    E_WINDOW = tuple(args.ewin)
    _NTHREADS = args.nthreads

    # --- [1] Load Reference Bulk Structure ---
    print("\n--- [1] Loading Reference Bulk Structure ---")
    struct = load_structure(BULK_CIF)
    struct = to_primitive(struct)
    sga = SpacegroupAnalyzer(struct)
    struct_conv = sga.get_conventional_standard_structure()
    print_lattice_info("user‑cell", struct_conv.lattice.matrix)
    print_kpath_info(struct_conv, "bulk CIF ")
    print(f"  Loaded reference: {struct_conv.formula}")

    # --- [2] Load QD ---
    print("\n--- [2] Reading QD Structure (XYZ) and Aligning Axes ---")
    syms, coords = read_xyz(XYZ_PATH)
    print(f"  Read {len(syms)} atoms from '{XYZ_PATH}'.")
    bulk_B = struct.lattice.reciprocal_lattice.matrix  # NumPy array (Å⁻¹)

    # --- [3] Loading Lattice from User -----------------------------------------
    print("\n--- [3] Loading user‑defined lattice ---")
    user_lattice = LATTICE 
    syms_bulk_dft, coords_bulk_dft = read_xyz(BULK_DFT_XYZ_PATH)
 
    lat = Lattice(user_lattice)
    struct_dft = Structure(
        lattice=lat,
        species=syms_bulk_dft,               # from read_xyz()
        coords=coords_bulk_dft,              # cartesian Å
        coords_are_cartesian=True,  # important!
    )
    struct_dft = to_primitive(struct_dft)
    sga = SpacegroupAnalyzer(struct_dft)
    struct_dft_conv = sga.get_conventional_standard_structure()
    print(f"  Built struct_dft with {len(struct_dft_conv)} atoms")
    print_lattice_info("user‑cell", struct_dft_conv.lattice.matrix)
    print_kpath_info(struct_dft_conv, "bulk CIF ")

    # --- [4] (replace your old 'struct_rot' etc. if needed) --------------------
    kpts_cart_qd, labels_qd = get_kpoints_cart(struct_dft_conv, line_density=600)
    # Print some k-path points for debugging
    print("\n--- [5] QD-adapted k-path points (Å⁻¹) ---")
    for i, (k, lbl) in enumerate(zip(kpts_cart_qd, labels_qd)):
        if lbl:
            print(f"  [{i:4d}] {lbl:3s}: k = [{k[0]:.4f}, {k[1]:.4f}, {k[2]:.4f}] Å⁻¹")

    # (Optional) Compute k-path distances for plotting
    def k_path_distance(kpts):
        dk = np.linalg.norm(np.diff(kpts, axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(dk)))

    k_path = k_path_distance(kpts_cart_qd)
    print("\n  K-path distances (Å⁻¹):", k_path)

    # --- [5] AO Fourier Transform ---
    print("\n--- [5] AO Fourier Transform ---")
    kpts_bohr = kpts_cart_qd / _BOHR_PER_ANG
    print(f"  kpts_bohr shape: {kpts_bohr.shape}")

    basis_dict = parse_basis(BASIS_TXT, BASIS_NAME)
    shell_dicts_cartesian = build_shell_dicts(syms, coords, basis_dict)
    shell_dicts_spherical = [{**sh, 'pure': True} for sh in shell_dicts_cartesian]
    n_ao = sum(2 * sh['l'] + 1 for sh in shell_dicts_spherical)
    print(f"  Using {n_ao} spherical AOs for QD.")

    print("\nReading MOs from coefficients file")
    C, eps_Ha, occ = read_mos_txt2(MO_PATH, n_ao)
    print("  idx      Energy (Ha)      Energy (eV)     Occupation")
    print("--------------------------------------------------------")
    for i, (eh, occ_i) in enumerate(zip(eps_Ha, occ)):
        print(f"{i+1:4d}   {eh:14.6f}   {eh*27.2114:12.6f}   {occ_i:8.4f}")
    print("\n .... done")

    print(f"\n  Running libint_fuzzy.ao_ft ...")
    F = libint_fuzzy.ao_ft(shell_dicts_spherical, kpts_bohr, nthreads=_NTHREADS)
    print(f"  AO FT result: F.shape = {F.shape} (n_ao, n_k)")

    Psi_k = C.T @ F   # (n_mo, n_k)
    print(f"  Psi_k shape: {Psi_k.shape} (n_mo, n_k)")
    intensity = np.abs(Psi_k)**2
    print(f"  Intensity matrix shape: {intensity.shape}")

    # --- [6] Plotting ---
    print("\n--- [6] Plotting Fuzzy Band Structure ---")
    fuzzy_band_imshow(
        kpts_cart_qd, labels_qd,
        eps_Ha, occ, intensity,
        dE=0.01,
        energy_window=E_WINDOW,
        vmin_scale=1e3,
        blur_sigma=None,
        outfile="fuzzy_band.png"
    )
    print("Done. Plot saved to fuzzy_band.png")

    # Plot intensity for selected MOs (HOMO+1, LUMO)
    homo = int(np.where(occ > 0)[0][-1])
    print("\n--- [7] Plotting Selected MO Intensity Profiles ---")
    plot_mo_intensity(
        kpts_cart_qd, labels_qd, k_path,
        intensity, eps_Ha,
        mo_indices=[homo, homo + 1],  # Plot HOMO and LUMO
        gamma_norm=0.3
    )

    print("\nAll steps complete. Inspect your fuzzy_band.png for sharpness and k-label alignment!")

if __name__ == "__main__":
    main() 

