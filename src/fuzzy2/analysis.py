from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import eigh
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

# ------------------ AO helpers ------------------

def shells_to_ao_arrays(shells):
    """
    Expand per-shell metadata to per-AO arrays.
    Returns:
      atom_idx_ao : (nAO,) integer atom index per AO
      sym_ao      : (nAO,) atom symbol per AO
      l_ao        : (nAO,) angular momentum number per AO (0,1,2,...)
    """
    atom_idx_list, sym_list, l_list = [], [], []
    for sh in shells:
        l = int(sh["l"])
        mult = 2 * l + 1
        atom_idx_list.extend([int(sh["atom_idx"])] * mult)
        sym_list.extend([sh["sym"]] * mult)
        l_list.extend([l] * mult)
    return (
        np.asarray(atom_idx_list, dtype=int),
        np.asarray(sym_list, dtype=object),
        np.asarray(l_list, dtype=int),
    )

# ------------------ Core building blocks ------------------

def compute_pdos_weights(C, S, method="mulliken"):
    """
    Per-AO per-MO weights for PDOS.
    Mulliken:  q_mu(n) = C_{mu n} (S C)_{mu n}
    Löwdin:    |C'|^2 with C' = S^{-1/2} C
    C: (nAO, nMO), S: (nAO, nAO)
    Returns: (nAO, nMO)
    """
    # --- Ensure columns of C are S-orthonormal: c_k^H S c_k = 1 ---
    norms = (C.conj() * (S @ C)).sum(axis=0).real
    scale = 1.0 / np.sqrt(np.maximum(norms, 1e-12))
    Cn = C * scale  # column-wise scaled copy

    if method == "lowdin":
        s_vals, s_vecs = eigh(S)
        s_vals = np.maximum(s_vals, 1e-12)  # guard against tiny eigenvalues
        S_inv_sqrt = s_vecs @ np.diag(s_vals**-0.5) @ s_vecs.T
        C_prime = S_inv_sqrt @ Cn
        pdos_weights = np.abs(C_prime) ** 2
        which = "Löwdin"
    else:
        # Mulliken with S-orthonormalized MOs
        pdos_weights = (Cn.conj() * (S @ Cn)).real
        which = "Mulliken"
    print(f"  ✓ PDOS weights computed using {which} population analysis.")
    return pdos_weights

def project_pdos(pdos_weights, shells):
    """
    Sum AO-resolved weights into atoms (vectorized).
    shells: list of per-shell dicts; expanded internally to per-AO.
    Returns (nAtoms, nMO)
    """
    atom_idx_ao, _, _ = shells_to_ao_arrays(shells)
    n_atoms = int(atom_idx_ao.max()) + 1 if atom_idx_ao.size else 0
    projected = np.zeros((n_atoms, pdos_weights.shape[1]), dtype=pdos_weights.dtype)
    np.add.at(projected, atom_idx_ao, pdos_weights)
    print("  ✓ PDOS projected onto atoms.")
    return projected

# ------------------ DOS / broadening utilities ------------------

def gaussian_kernel(E, eps, sigma):
    X = (E[:, None] - eps[None, :]) / sigma
    return np.exp(-0.5 * X * X) / (sigma * np.sqrt(2 * np.pi))

def broaden(values_per_mo, energies, energy_grid, sigma):
    """
    values_per_mo: (nMO,)
    energies:      (nMO,)
    energy_grid:   (nE,)
    """
    G = gaussian_kernel(energy_grid, energies, sigma)  # (nE, nMO)
    return (values_per_mo * G).sum(axis=1)

def compute_dos_states(eps, energy_grid, sigma):
    """Count-of-states DOS (occupied + unoccupied, each MO weight=1)."""
    ones = np.ones_like(eps)
    return broaden(ones, eps, energy_grid, sigma)

def compute_dos_electrons(eps, occ, energy_grid, sigma):
    """Electron DOS (weights by occupations)."""
    w = np.asarray(occ) if occ is not None else 1.0
    return broaden(w, eps, energy_grid, sigma)

def fermi_from_occ(eps, occ, ewin=None, tol=1e-8):
    """
    Midgap Fermi (HOMO/LUMO midpoint). If occ not available, use center of ewin.
    """
    if occ is not None and np.size(occ) and np.any(np.asarray(occ) > tol):
        filled = np.where(np.asarray(occ) > tol)[0]
        h = filled[-1]
        if h + 1 < eps.size:
            return 0.5 * (eps[h] + eps[h + 1])
    if ewin is not None:
        return 0.5 * (ewin[0] + ewin[1])
    return float(np.median(eps))

# ------------------ Compact, one-line analyses ------------------

def _fmt0(x, eps=1e-6, nd=3):
    """Pretty float with small values shown as +0.000 (no -0.000)."""
    v = 0.0 if abs(x) < eps else x
    return f"{v:.{nd}f}"

def print_pdos_population_analysis(pdos_weights, shells, eps, occ, ewin=None,
                                   spd_thresh=0.01, include_unocc=True):
    """
    Compact per-MO one-liners inside the energy window:
    MO <n> E=<eV> occ=<o> | <sym>=<tot> [s:<v> p:<v> d:<v> ; ...] ; <sym2>=...
    Only s/p/d parts with contribution >= spd_thresh are printed.
    include_unocc=True prints both occupied and LUMO-side MOs within ewin.
    """
    print("\n--- [PDOS Population Analysis] ---")
    norb = pdos_weights.shape[1]
    _, sym_ao, l_ao = shells_to_ao_arrays(shells)

    # Precompute AO index groups
    sym_to_idx = defaultdict(list)
    sym_l_to_idx = defaultdict(list)
    for i, (sym, l) in enumerate(zip(sym_ao, l_ao)):
        sym_to_idx[sym].append(i)
        sym_l_to_idx[(sym, l)].append(i)

    symbols = sorted(sym_to_idx.keys())

    for n in range(norb):
        if ewin is not None and not (ewin[0] <= eps[n] <= ewin[1]):
            continue
        if not include_unocc and occ is not None and occ[n] < 1e-8:
            continue
        parts = []
        for sym in symbols:
            idx_all = sym_to_idx[sym]
            tot = pdos_weights[idx_all, n].sum()
            # s/p/d parts filtered by threshold
            spd_parts = []
            for l, tag in [(0, "s"), (1, "p"), (2, "d")]:
                idx = sym_l_to_idx.get((sym, l), [])
                if idx:
                    val = pdos_weights[idx, n].sum()
                    if val >= spd_thresh:
                        spd_parts.append(f"{tag}:{_fmt0(val)}")
            parts.append(f"{sym}={_fmt0(tot, nd=4)} [{', '.join(spd_parts)}]")
        occ_str = f"{occ[n]:.1f}" if occ is not None else "NA"
        print(f"MO {n:4d}  E={eps[n]:8.3f} eV  occ={occ_str}  |  " + "  ;  ".join(parts))

def compute_coop(C, S, shells, atom_pairs):
    """
    COOP weights per MO for each atom-type pair (e.g., 'Cd-Se').
    Efficient and correct: precompute XB[b] = S[:, B] @ C[B, :] and then
    COOP_AB(n) = sum_{i in A} C_{i n} * XB[b][i, n]   (×2 for Mulliken).
    Returns dict: pair -> (nMO,)
    """
    # AO indices per symbol
    _, sym_ao, _ = shells_to_ao_arrays(shells)
    symbol_to_idx = {sym: np.where(sym_ao == sym)[0] for sym in set(sym_ao)}

    # Precompute XB for each 'B' symbol only once
    XB = {}
    for sym_b, idx_b in symbol_to_idx.items():
        if idx_b.size:
            XB[sym_b] = S[:, idx_b] @ C[idx_b, :]  # (nAO, nMO)
        else:
            XB[sym_b] = None

    results = {}
    for pair in atom_pairs:
        a, b = pair.split("-")
        idx_a = symbol_to_idx.get(a, np.array([], dtype=int))
        XB_b  = XB.get(b, None)
        if idx_a.size == 0 or XB_b is None:
            results[pair] = np.zeros(C.shape[1], dtype=float)
            continue
        # Now both arrays are (|A|, nMO) → elementwise multiply then sum rows
        coop_n = (C[idx_a, :] * XB_b[idx_a, :]).sum(axis=0).real * 2.0
        results[pair] = coop_n
    print(f"  ✓ COOP computed for {len(atom_pairs)} atom pairs.")
    return results

def print_coop_analysis(coop_weights, eps, occ, ewin=None, include_unocc=True):
    """
    Compact per-MO line inside ewin:
    MO <n> E=<eV> occ=<o> | pair1=<v> ; pair2=<v> ; ...
    """
    print("\n--- [COOP Analysis] ---")
    norb = len(eps)
    pairs = sorted(coop_weights.keys())
    for n in range(norb):
        if ewin is not None and not (ewin[0] <= eps[n] <= ewin[1]):
            continue
        if not include_unocc and occ is not None and occ[n] < 1e-8:
            continue
        parts = [f"{p}={_fmt0(coop_weights[p][n], nd=4)}" for p in pairs]
        occ_str = f"{occ[n]:.1f}" if occ is not None else "NA"
        print(f"MO {n:4d}  E={eps[n]:8.3f} eV  occ={occ_str}  |  " + "  ;  ".join(parts))

# ------------------ Helper utilities for plotting ------------------

@dataclass
class FuzzyMapData:
    centres: np.ndarray
    intensity_grid: np.ndarray
    labels: List[str]
    tick_positions: List[float]
    tick_labels: List[str]
    extent: Tuple[float, float, float, float]
    norm: Normalize
    cmap: plt.Colormap
    energy_window: Tuple[float, float]


def _sym_to_atom_indices(shells) -> Dict[str, np.ndarray]:
    atom_idx_shell = np.array([int(sh["atom_idx"]) for sh in shells], dtype=int)
    sym_shell = np.array([sh["sym"] for sh in shells], dtype=object)
    return {
        sym: np.unique(atom_idx_shell[sym_shell == sym])
        for sym in set(sym_shell)
    }


def _prepare_pdos_data(C, S, shells, method="mulliken"):
    pdos_weights = compute_pdos_weights(C, S, method=method)
    projected = project_pdos(pdos_weights, shells)
    sym_to_atom_indices = _sym_to_atom_indices(shells)
    return pdos_weights, projected, sym_to_atom_indices


def _pdos_curves_for_symbols(projected, sym_to_atom_indices, pdos_atom_list,
                             eps, energy_axis, sigma):
    curves, labels = [], []
    for sym in pdos_atom_list:
        rows = sym_to_atom_indices.get(sym, np.array([], dtype=int))
        if rows.size == 0:
            continue
        weights_mo = projected[rows, :].sum(axis=0)
        curves.append(broaden(weights_mo, eps, energy_axis, sigma))
        labels.append(sym)
    if not curves:
        return [], [], None
    Y = np.column_stack(curves)
    return curves, labels, np.cumsum(Y, axis=1)


def _prepare_fuzzy_map(kpts_cart, labels, intensity, eps_eV, ewin,
                       sigma_ev, scaled_vmin):
    from .plotting import dedup_kpath_strict, _plain_k_label, _fade_cmap

    _, labels_dedup, keep = dedup_kpath_strict(np.asarray(kpts_cart), labels)
    I = np.asarray(intensity)[:, keep]

    E = np.asarray(eps_eV)
    window_mask = (E >= ewin[0] - 4 * sigma_ev) & (E <= ewin[1] + 4 * sigma_ev)
    E = E[window_mask]
    I = I[window_mask, :]

    dE = max(0.5 * sigma_ev, 0.01)
    edges = np.arange(ewin[0], ewin[1] + dE, dE)
    centres = 0.5 * (edges[:-1] + edges[1:])
    Z = np.zeros((centres.size, I.shape[1]), dtype=float)
    for En, Ik in zip(E, I):
        w = np.exp(-0.5 * ((centres - En) / sigma_ev) ** 2)
        Z += np.outer(w, Ik)

    vmax = np.percentile(Z, 99.9) if np.any(Z > 0) else 1.0
    pos = Z[Z > 0]
    vmin = np.percentile(pos, 5) if pos.size else 1e-6
    vmin_eff = max(vmin, vmax / float(scaled_vmin)) if scaled_vmin is not None else vmin
    if vmin_eff >= vmax:
        vmin_eff = max(vmax * 0.5, 1e-8)
    norm = LogNorm(vmin=vmin_eff, vmax=vmax)
    cmap = _fade_cmap()
    cmap.set_bad('black')
    cmap.set_under('black')

    tick_pos, tick_lab, prev = [], [], None
    for i, lbl in enumerate(labels_dedup):
        if lbl and lbl != prev:
            tick_pos.append(float(i))
            tick_lab.append(_plain_k_label(lbl))
            prev = lbl

    extent = (0.0, float(Z.shape[1] - 1), float(centres.min()), float(centres.max()))
    return FuzzyMapData(
        centres=centres,
        intensity_grid=Z,
        labels=list(labels_dedup),
        tick_positions=tick_pos,
        tick_labels=tick_lab,
        extent=extent,
        norm=norm,
        cmap=cmap,
        energy_window=(float(ewin[0]), float(ewin[1])),
    )


# ------------------ Bond detection + colors ------------------

# Fallback covalent radii in Å (used if periodictable is not available)
_COV_FALLBACK = {
    "H":0.31,"C":0.76,"N":0.71,"O":0.66,"F":0.57,"Cl":0.99,"Br":1.14,"I":1.33,
    "Na":1.66,"K":2.03,"Rb":2.16,"Cs":2.35,
    "Cd":1.44,"Hg":1.32,"Te":1.38,"Pb":1.44,"Se":1.20
}

def covalent_radius(sym: str) -> float:
    try:
        import periodictable as pt
        return float(pt.elements.symbol(sym).covalent_radius)
    except Exception:
        return _COV_FALLBACK.get(sym, 1.2)

def bonded_symbol_pairs_from_overlap(S, shells, s_thresh: float = 0.05):
    """
    Infer which symbol pairs (e.g., 'Hg-Te') are directly bonded by testing if
    any AO pair across two symbols has |S_ij| >= s_thresh.
    """
    _, sym_ao, _ = shells_to_ao_arrays(shells)
    sym_to_idx = {sym: np.where(sym_ao == sym)[0] for sym in set(sym_ao)}
    symbols = sorted(sym_to_idx.keys())

    bonded = set()
    for i, a in enumerate(symbols):
        Ai = sym_to_idx[a]
        for b in symbols[i+1:]:
            Bi = sym_to_idx[b]
            if Ai.size == 0 or Bi.size == 0:
                continue
            Sab = S[np.ix_(Ai, Bi)]
            if Sab.size and (np.abs(Sab) >= s_thresh).any():
                bonded.add("-".join(sorted((a,b))))
    return sorted(bonded)

def bonded_symbol_pairs_from_distance(atom_symbols, positions, scale: float = 1.15):
    """
    Return symbol pairs that are bonded by distance criterion:
    min distance between any atom of A and B < scale*(rcov(A)+rcov(B)).
    """
    import numpy as np
    atom_symbols = np.asarray(atom_symbols, dtype=object)
    R = np.asarray(positions, float)
    n = len(atom_symbols)

    # group atom indices by symbol
    from collections import defaultdict
    idx_by_sym = defaultdict(list)
    for i, s in enumerate(atom_symbols):
        idx_by_sym[str(s)].append(i)
    syms = sorted(idx_by_sym.keys())

    bonded = set()
    for i, a in enumerate(syms):
        Ai = np.array(idx_by_sym[a], int)
        for b in syms[i+1:]:
            Bi = np.array(idx_by_sym[b], int)
            if Ai.size == 0 or Bi.size == 0: continue
            Ra, Rb = R[Ai], R[Bi]
            # pairwise distances
            d2 = np.sum((Ra[:,None,:]-Rb[None,:,:])**2, axis=2)
            dmin = float(np.sqrt(d2.min()))
            cutoff = scale*(covalent_radius(a) + covalent_radius(b))
            if dmin < cutoff:
                bonded.add("-".join(sorted((a,b))))
    return sorted(bonded)

def pair_color_map(pairs: List[str]):
    """
    Deterministic color assignment for a given ordered list of pairs,
    using Matplotlib's current prop cycle so standalone and combo match.
    """
    cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ["#4c72b0","#dd8452","#55a868","#c44e52","#8172b3","#937860"])
    cmap = {}
    for i, p in enumerate(pairs):
        cmap[p] = cycle[i % len(cycle)]
    return cmap
# ------------------ Plot wrappers (original workflow preserved) ------------------

def plot_dos_and_pdos(eps, occ, C, S, shells, pdos_atom_list, ewin,
                      method="mulliken", sigma=0.08):
    """
    Make 'dos_pdos.png' with cumulative, filled PDOS (per atom type)
    overlaid with a thin line for total count-of-states DOS.
    Uses sigma=0.08 by default (independent of fuzzy sigma).
    Also draws a vertical line at the midgap Fermi level.
    """
    print("\n--- [Plotting DOS/PDOS] ---")
    energy_grid = np.linspace(ewin[0], ewin[1], 1000)

    # Build PDOS curves
# PDOS weights and atom-projected (nAtoms, nMO)
    _, projected, sym_to_atom_indices = _prepare_pdos_data(C, S, shells, method=method)

    # Build per-symbol PDOS curves (NO occupation weighting → include LUMOs)
    curves, labels, Ycum = _pdos_curves_for_symbols(
        projected, sym_to_atom_indices, pdos_atom_list, eps, energy_grid, sigma
    )

    if not curves:
        print("  ⚠︎ No PDOS curves to plot (empty pdos_atom_list or no matching atoms).")
        return

    if Ycum is None:
        Y = np.column_stack(curves)
        Ycum = np.cumsum(Y, axis=1)
    zero = np.zeros_like(energy_grid)

    plt.figure(figsize=(7.0, 4.4))
    # Fill stacked PDOS
    prev = zero
    for j in range(Ycum.shape[1]):
        plt.fill_between(energy_grid, prev, Ycum[:, j], alpha=0.75, linewidth=0.0, label=labels[j])
        prev = Ycum[:, j]
    # Overlay thin total DOS
    
    # Black "Total DOS" as the sum of the plotted PDOS curves (ensures identity)
    total_curve = Ycum[:, -1] if Ycum is not None and Ycum.ndim == 2 else np.zeros_like(energy_grid)
    plt.plot(energy_grid, total_curve, lw=1.1, color="k", alpha=0.9, label="Total DOS")

    # Midgap (Fermi-like) vertical line
    ef = fermi_from_occ(eps, occ, ewin)
    plt.axvline(ef, color="k", ls="--", lw=1.0, alpha=0.8)
    plt.text(ef, 0.98 * float(max(Ycum.max(), total_curve.max() if np.ndim(total_curve)>0 else total_curve)), " $E_F$", ha="left", va="top")

    plt.xlim(ewin[0], ewin[1])
    ymax = max(Ycum.max(), (total_curve.max() if np.ndim(total_curve)>0 else float(total_curve))) * 1.05
    plt.ylim(0.0, ymax if ymax > 0 else 1.0)
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS (a.u.)")
    plt.title("Stacked PDOS (cumulative, filled) + total DOS")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dos_pdos.png", dpi=150)
    print("  ✓ DOS/PDOS plot saved as dos_pdos.png")

def plot_coop(eps, C, S, shells, coop_pair_list, ewin,
              method="mulliken", sigma=0.1,
              normalize=True, bond_mode="distance", s_thresh=0.05,
              atom_symbols=None, positions=None, bond_scale=1.05):
    """
    Make 'coop.png' with COOP curves for selected pairs (filled pos/neg lobes).
    If normalize=True, scales to [-1, 1].
    If bond_mode in {"overlap","distance"}, filter pairs to actually bonded.
      - "overlap": uses AO overlap threshold s_thresh.
      - "distance": requires atom_symbols + positions to detect bonds by covalent radii.
    If coop_pair_list is None, discover all bonded pairs under the chosen mode.
    """
    print("\n--- [Plotting COOP] ---")
    energy_grid = np.linspace(ewin[0], ewin[1], 1000)

    # Decide which pairs to plot
    def canon(p):
        a,b = [t.strip() for t in p.split("-")]
        return "-".join(sorted((a,b)))

    if bond_mode == "distance" and (atom_symbols is not None and positions is not None):
        bonded = set(bonded_symbol_pairs_from_distance(atom_symbols, positions, scale=bond_scale))
    elif bond_mode == "overlap":
        bonded = set(bonded_symbol_pairs_from_overlap(S, shells, s_thresh=s_thresh))
    else:
        bonded = None  # accept all

    if coop_pair_list is None:
        pairs = sorted(bonded) if bonded is not None else []
    else:
        pairs = []
        seen=set()
        for p in coop_pair_list:
            cp = canon(p)
            if bonded is not None and cp not in bonded:
                continue
            if cp not in seen:
                seen.add(cp); pairs.append(cp)

    coop_weights = compute_coop(C, S, shells, pairs)



    # Colors shared between standalone and combo
    cmap = pair_color_map(pairs)

    plt.figure(figsize=(7.0, 4.4))
    curves = {}
    for pair in pairs:
        vals = coop_weights.get(pair)
        if vals is None: 
            continue
        curve = broaden(vals, eps, energy_grid, sigma)
        curves[pair] = curve

    # Normalize to [-1,1] if requested
    if normalize and curves:
        m = max(float(np.max(np.abs(v))) for v in curves.values() if v.size)
        if m > 0:
            for k in list(curves.keys()):
                curves[k] = np.clip(curves[k]/m, -1.0, 1.0)

    for pair, curve in curves.items():
        pos = np.clip(curve, 0, None)
        neg = np.clip(curve, None, 0)
        col = cmap.get(pair, None)
        plt.fill_between(energy_grid, 0, pos, alpha=0.5, label=f"{pair} (+)", color=col)
        plt.fill_between(energy_grid, 0, neg, alpha=0.5, label=f"{pair} (−)", color=col)

    plt.axhline(0.0, lw=0.8, color="k", alpha=0.6)
    plt.xlim(ewin[0], ewin[1])
    if normalize:
        plt.ylim(-1.05, 1.05)
        plt.ylabel("COOP (normalized)")
    else:
        plt.ylabel("COOP (a.u.)")
    plt.xlabel("Energy (eV)")
    plt.title("Crystal Orbital Overlap Population")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("coop.png", dpi=150)
    print("  ✓ COOP plot saved as coop.png")
# ------------------ Combined fuzzy+PDOS (kept for convenience) ------------------

def plot_fuzzy_and_pdos_combo(
    kpts_cart,
    labels,
    k_path_dist,
    eps_eV,                 # energies in eV
    intensity,              # (nMO, nK)
    C, S, shells,
    pdos_atom_list,
    ewin,
    sigma_ev=0.10,          # fuzzy broadening
    sigma_pdos=0.08,        # PDOS broadening
    midgap=None,
    occ=None,
    outfile="fuzzy_plus_pdos.png",
    scaled_vmin=1e-4,       # match your CLI/standalone fuzzy
):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fuzzy = _prepare_fuzzy_map(
        kpts_cart, labels, intensity, eps_eV, ewin, sigma_ev, scaled_vmin
    )

    _, projected, sym_to_atom = _prepare_pdos_data(C, S, shells, method="mulliken")
    curves, labels_p, Ycum = _pdos_curves_for_symbols(
        projected, sym_to_atom, pdos_atom_list, eps_eV, fuzzy.centres, sigma_pdos
    )

    xmax = 1.0
    if Ycum is not None and Ycum.size:
        xmax = float(Ycum.max()) * 1.05

    ef = (fermi_from_occ(eps_eV, occ, ewin) if midgap is None else float(midgap))

    fig = plt.figure(figsize=(9.6, 5.4), constrained_layout=False)
    gs = GridSpec(1, 100, figure=fig, wspace=0.05)
    axL = fig.add_subplot(gs[0, :85])
    axR = fig.add_subplot(gs[0, 85:97], sharey=axL)
    infoS = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 97:], height_ratios=[3, 2], hspace=0.15)
    axLeg = fig.add_subplot(infoS[0])
    axC = fig.add_subplot(infoS[1])

    extent = [fuzzy.extent[0], fuzzy.extent[1], fuzzy.extent[2], fuzzy.extent[3]]
    im = axL.imshow(
        fuzzy.intensity_grid,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap=fuzzy.cmap,
        norm=fuzzy.norm,
    )
    axL.set_facecolor('black')
    axL.set_ylim(fuzzy.energy_window)
    axL.set_ylabel("Energy (eV)")
    axL.set_xlabel("High-Symmetry k-Path")
    prev_label = None
    for i, lbl in enumerate(fuzzy.labels):
        if lbl and lbl != prev_label:
            axL.axvline(i, color='gray', lw=0.5, alpha=0.6)
            prev_label = lbl
    axL.set_xticks(fuzzy.tick_positions)
    axL.set_xticklabels(fuzzy.tick_labels)
    axL.axhline(ef, color='w', ls='--', lw=1.2, alpha=0.9)

    axR.tick_params(left=False, labelleft=False)
    axR.spines["left"].set_visible(False)
    handles = []
    if Ycum is not None:
        prev = np.zeros_like(fuzzy.centres)
        for j in range(Ycum.shape[1]):
            p = axR.fill_betweenx(
                fuzzy.centres, prev, Ycum[:, j], alpha=0.75, linewidth=0.0, label=labels_p[j]
            )
            handles.append(p)
            prev = Ycum[:, j]
        axR.set_xlim(0.0, xmax)
    else:
        axR.set_xlim(0.0, 1.0)
    axR.set_xlabel("PDOS (a.u.)")

    axLeg.axis("off")
    if handles:
        axLeg.legend(handles, labels_p, loc="upper left", frameon=False, fontsize=9)

    cb = plt.colorbar(im, cax=axC, orientation='vertical')
    cb.set_label("Intensity (arb., log scale)", fontsize=9)

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"✓ Combined fuzzy+PDOS plot saved to {outfile}")
    plt.close(fig)

# ------------------ Combined fuzzy+PDOS+COOP + CSV exports ------------------
def plot_fuzzy_pdos_coop_combo(
    kpts_cart, labels, k_path_dist,
    eps_eV, intensity, C, S, shells,
    pdos_atom_list, coop_pair_list, ewin,
    sigma_ev=0.10, sigma_pdos=0.08,       # COOP uses sticks (no σ)
    midgap=None, occ=None,
    outfile="fuzzy_pdos_coop.png",
    scaled_vmin=1e-4,
):
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D

    fuzzy = _prepare_fuzzy_map(
        kpts_cart, labels, intensity, eps_eV, ewin, sigma_ev, scaled_vmin
    )
    ncol = fuzzy.intensity_grid.shape[1]

    _, projected, sym_to_atom = _prepare_pdos_data(C, S, shells, method="mulliken")
    curves_p, labels_p, Ycum = _pdos_curves_for_symbols(
        projected, sym_to_atom, pdos_atom_list, eps_eV, fuzzy.centres, sigma_pdos
    )

    
    # -------- COOP: select bonded pairs (distance preferred if data is available) --------
    def _canonical_pair(p):
        a, b = [s.strip() for s in p.split("-")]
        return "-".join(sorted((a, b)))

    # Attempt distance-based bonding if atom symbols/positions are attached to shells.
    atom_symbols_opt = getattr(shells, "atom_symbols", None) if hasattr(shells, "__dict__") else None
    positions_opt     = getattr(shells, "positions", None) if hasattr(shells, "__dict__") else None

    if coop_pair_list is None or len(coop_pair_list) == 0:
        if atom_symbols_opt is not None and positions_opt is not None:
            pairs_canon = bonded_symbol_pairs_from_distance(atom_symbols_opt, positions_opt, scale=1.15)
        else:
            pairs_canon = bonded_symbol_pairs_from_overlap(S, shells, s_thresh=0.05)
    else:
        raw = [_canonical_pair(p) for p in coop_pair_list if "-" in p]
        bonded = set(bonded_symbol_pairs_from_overlap(S, shells, s_thresh=0.05))
        # If distance info available, refine with distance
        if atom_symbols_opt is not None and positions_opt is not None:
            bonded = bonded.intersection(set(bonded_symbol_pairs_from_distance(atom_symbols_opt, positions_opt, scale=1.15)))
        pairs_canon = [p for p in raw if p in bonded]

    # deduplicate while preserving order
    pairs_unique = []
    seen = set()
    for p in pairs_canon:
        if p not in seen:
            seen.add(p); pairs_unique.append(p)

    coop_weights = compute_coop(C, S, shells, pairs_unique)
    # --- COOP energy window + normalization + colors ---
    # 1) pick MO energies inside the plotting window
    E_all = np.asarray(eps_eV)
    mask_sticks = (E_all >= ewin[0]) & (E_all <= ewin[1])
    Ener_sticks = E_all[mask_sticks] if E_all.size else np.asarray([], dtype=float)

    # 2) global normalization so sticks are in [-1, 1]
    global_max = 0.0
    for p in pairs_unique:
        vals = np.asarray(coop_weights.get(p))
        if vals is None or vals.size == 0:
            continue
        v = vals[mask_sticks]
        if v.size:
            global_max = max(global_max, float(np.max(np.abs(v))))
    scale_norm = (1.0 / global_max) if global_max > 0 else 1.0

    # 3) color map for pairs (same palette as standalone)
    palette = plt.rcParams['axes.prop_cycle'].by_key().get(
        'color',
        ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
    )
    coop_color_map = {p: palette[i % len(palette)] for i, p in enumerate(pairs_unique)}


    ef = (fermi_from_occ(eps_eV, occ, ewin) if midgap is None else float(midgap))

    fig = plt.figure(figsize=(12, 5.4), constrained_layout=False)
    gs = GridSpec(1, 120, figure=fig, wspace=0.05)
    axF = fig.add_subplot(gs[0, :70])
    axP = fig.add_subplot(gs[0, 70:85], sharey=axF)
    axC = fig.add_subplot(gs[0, 85:100], sharey=axF)
    infoS = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 100:], height_ratios=[3, 2], hspace=0.15)
    axLeg = fig.add_subplot(infoS[0])
    axBar = fig.add_subplot(infoS[1])

    extent = [fuzzy.extent[0], fuzzy.extent[1], fuzzy.extent[2], fuzzy.extent[3]]
    im = axF.imshow(
        fuzzy.intensity_grid,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap=fuzzy.cmap,
        norm=fuzzy.norm,
    )
    axF.set_facecolor('black')
    axF.set_ylim(fuzzy.energy_window)
    axF.set_ylabel("Energy (eV)")
    axF.set_xlabel("High-Symmetry k-Path")
    prev_label = None
    for i, lbl in enumerate(fuzzy.labels):
        if lbl and lbl != prev_label:
            axF.axvline(i, color='gray', lw=0.5, alpha=0.6)
            prev_label = lbl
    axF.set_xticks(fuzzy.tick_positions)
    axF.set_xticklabels(fuzzy.tick_labels)
    axF.axhline(ef, color='w', ls='--', lw=1.2, alpha=0.9)

    axP.tick_params(left=False, labelleft=False)
    axP.spines["left"].set_visible(False)
    handles_p = []
    if Ycum is not None:
        prev = np.zeros_like(fuzzy.centres)
        for j in range(Ycum.shape[1]):
            p = axP.fill_betweenx(
                fuzzy.centres, prev, Ycum[:, j], alpha=0.75, linewidth=0.0, label=labels_p[j]
            )
            handles_p.append(p)
            prev = Ycum[:, j]
        axP.set_xlim(0, float(Ycum.max()) * 1.05)
    else:
        axP.set_xlim(0, 1)
    axP.set_xlabel("PDOS")

    # COOP sticks: plot both allowed pairs with semi-transparency; weakest → strongest
    axC.tick_params(left=False, labelleft=False); axC.spines["left"].set_visible(False)
    axC.axvline(0, lw=0.8, color="k", alpha=0.7); axC.set_xlabel("COOP")

    # order by max magnitude to mitigate overdraw
    pair_max = {p: (np.max(np.abs(np.asarray(coop_weights.get(p, [0]))[mask_sticks]))
                    if Ener_sticks.size else 0.0)
                for p in pairs_unique}
    order = sorted(pairs_unique, key=lambda p: pair_max[p])  # weakest first

    coop_handles = []
    x_max = 0.0
    for p in order:
        vals = np.asarray(coop_weights.get(p))
        if Ener_sticks.size == 0 or vals.size == 0:
            continue
        vV = np.clip(scale_norm * vals[mask_sticks], -1.0, 1.0)
        col = coop_color_map.get(p, "#555555")
        axC.hlines(Ener_sticks, 0.0, vV, colors=col, linewidths=1.1, alpha=0.65)
        x_max = max(x_max, np.max(np.abs(vV)))
        coop_handles.append(Line2D([0], [0], color=col, lw=2, label=f"{p} (+/−)"))
    axC.set_xlim(-1.05, 1.05)

    # Legends + slimmer colorbar
    axLeg.axis("off")
    all_handles = handles_p + coop_handles
    all_labels  = ([h.get_label() for h in handles_p] +
                   [h.get_label() for h in coop_handles])
    if all_handles:
        axLeg.legend(all_handles, all_labels, loc="upper left", frameon=False, fontsize=9)

    cb = plt.colorbar(im, cax=axBar, orientation='vertical')
    cb.set_label("Intensity (arb., log scale)", fontsize=9)
    bbox = axBar.get_position()
    axBar.set_position([bbox.x0 + 0.35 * bbox.width, bbox.y0, 0.55 * bbox.width, bbox.height])

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"✓ Combined fuzzy+PDOS+COOP plot saved to {outfile}")
    plt.close(fig)

    # ---- CSV exports (fuzzy & PDOS unchanged; COOP exports sticks for filtered pairs) ----
    # fuzzy
    # --- Compact binary export (recommended) ---
    np.savez_compressed(
        "fuzzy_data.npz",
        centres=fuzzy.centres.astype(np.float16),
        intensity=fuzzy.intensity_grid.astype(np.float16),
        tick_positions=np.asarray(fuzzy.tick_positions, dtype=np.float16),
        tick_labels=np.asarray(fuzzy.tick_labels, dtype=object),
        labels=np.asarray(fuzzy.labels, dtype=object),
        extent=np.asarray(fuzzy.extent, dtype=np.float16),
        ewin=np.asarray(ewin, dtype=np.float16),
    )
    print("  ✓ Exported fuzzy → fuzzy_data.npz (compressed)")

    # pdos
    if Ycum is not None and len(labels_p) > 0:
        with open("pdos_data.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Energy_eV"] + labels_p)
            for iE, E in enumerate(fuzzy.centres):
                row = [E] + [Ycum[iE, j] for j in range(len(labels_p))]
                w.writerow(row)
        print("  ✓ Exported PDOS → pdos_data.csv")

    # coop sticks
    if len(pairs_unique) > 0 and Ener_sticks.size > 0:
        with open("coop_data.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["MO_Energy_eV"] + pairs_unique)
            cols = [np.asarray(coop_weights[p])[mask_sticks] for p in pairs_unique]
            for i in range(Ener_sticks.size):
                w.writerow([Ener_sticks[i]] + [cols[j][i] for j in range(len(cols))])
        print("  ✓ Exported COOP (sticks) → coop_data.csv")

