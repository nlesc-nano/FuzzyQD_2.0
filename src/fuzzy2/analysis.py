import numpy as np
from scipy.linalg import eigh
from collections import defaultdict
import matplotlib.pyplot as plt


# ------------------ AO helpers (expand shells -> AO arrays) ------------------

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
    if method == "lowdin":
        s_vals, s_vecs = eigh(S)
        S_inv_sqrt = s_vecs @ np.diag(s_vals**-0.5) @ s_vecs.T
        C_prime = S_inv_sqrt @ C
        pdos_weights = np.abs(C_prime) ** 2
        which = "Löwdin"
    else:
        pdos_weights = (C.conj() * (S @ C)).real
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


def print_pdos_population_analysis(pdos_weights, shells, eps, occ, ewin=None, spd_thresh=0.01, include_unocc=True):
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

    # Count-of-states DOS (shows LUMOs too)
    dos_states = compute_dos_states(eps, energy_grid, sigma)

    # PDOS weights and atom-projected (nAtoms, nMO)
    pdos_weights = compute_pdos_weights(C, S, method=method)
    projected = project_pdos(pdos_weights, shells)

    # Map: atom symbol -> unique atom indices (avoid double-counting same atom)
    atom_idx_shell = np.array([int(sh["atom_idx"]) for sh in shells], dtype=int)
    sym_shell      = np.array([sh["sym"] for sh in shells], dtype=object)
    sym_to_atom_indices = {
        sym: np.unique(atom_idx_shell[sym_shell == sym])
        for sym in set(sym_shell)
    }

    # Build per-symbol PDOS curves (NO occupation weighting → include LUMOs)
    curves, labels = [], []
    for sym in pdos_atom_list:
        atom_rows = sym_to_atom_indices.get(sym, np.array([], dtype=int))
        if atom_rows.size == 0:
            continue
        weights_mo = projected[atom_rows, :].sum(axis=0)  # (nMO,)
        curve = broaden(weights_mo, eps, energy_grid, sigma)  # (nE,)
        curves.append(curve)
        labels.append(sym)

    if not curves:
        print("  ⚠︎ No PDOS curves to plot (empty pdos_atom_list or no matching atoms).")
        return

    Y = np.column_stack(curves)            # (nE, nTypes)
    Ycum = np.cumsum(Y, axis=1)            # cumulative for stacked fill
    zero = np.zeros_like(energy_grid)

    plt.figure(figsize=(7.0, 4.4))
    # Fill stacked PDOS
    prev = zero
    for j in range(Ycum.shape[1]):
        plt.fill_between(energy_grid, prev, Ycum[:, j], alpha=0.75, linewidth=0.0, label=labels[j])
        prev = Ycum[:, j]
    # Overlay thin total DOS
    plt.plot(energy_grid, dos_states, lw=1.1, color="k", alpha=0.9, label="Total DOS")

    # Midgap (Fermi-like) vertical line
    ef = fermi_from_occ(eps, occ, ewin)
    plt.axvline(ef, color="k", ls="--", lw=1.0, alpha=0.8)
    plt.text(ef, 0.98 * max(Ycum.max(), dos_states.max()), " $E_F$", ha="left", va="top")

    plt.xlim(ewin[0], ewin[1])
    ymax = max(Ycum.max(), dos_states.max()) * 1.05
    plt.ylim(0.0, ymax if ymax > 0 else 1.0)
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS (a.u.)")
    plt.title("Stacked PDOS (cumulative, filled) + total DOS")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dos_pdos.png", dpi=150)
    print("  ✓ DOS/PDOS plot saved as dos_pdos.png")


def plot_coop(eps, C, S, shells, coop_pair_list, ewin,
              method="mulliken", sigma=0.1):
    """
    Make 'coop.png' with COOP curves for selected pairs (filled pos/neg lobes).
    """
    print("\n--- [Plotting COOP] ---")
    energy_grid = np.linspace(ewin[0], ewin[1], 1000)
    coop_weights = compute_coop(C, S, shells, coop_pair_list)

    plt.figure(figsize=(7.0, 4.4))
    for pair in coop_pair_list:
        vals = coop_weights.get(pair)
        if vals is None:
            continue
        curve = broaden(vals, eps, energy_grid, sigma)
        pos = np.clip(curve, 0, None)
        neg = np.clip(curve, None, 0)
        plt.fill_between(energy_grid, 0, pos, alpha=0.5, label=f"{pair} (+)")
        plt.fill_between(energy_grid, 0, neg, alpha=0.5, label=f"{pair} (−)")

    plt.axhline(0.0, lw=0.8, color="k", alpha=0.6)
    plt.xlim(ewin[0], ewin[1])
    plt.xlabel("Energy (eV)")
    plt.ylabel("COOP (a.u.)")
    plt.title("Crystal Orbital Overlap Population (filled)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("coop.png", dpi=150)
    print("  ✓ COOP plot saved as coop.png")

from matplotlib.colors import LogNorm, Normalize

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
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from .plotting import dedup_kpath_strict, _plain_k_label, _fade_cmap

    # ---- fuzzy grid (exactly like your standalone) ----
    kpts_dedup, labels_dedup, keep = dedup_kpath_strict(np.asarray(kpts_cart), labels)
    I = intensity[:, keep]
    ncol = I.shape[1]

    E = np.asarray(eps_eV)
    in_win = (E >= ewin[0] - 4*sigma_ev) & (E <= ewin[1] + 4*sigma_ev)
    E, I = E[in_win], I[in_win, :]

    dE = max(0.5*sigma_ev, 0.01)
    edges   = np.arange(ewin[0], ewin[1] + dE, dE)
    centres = 0.5 * (edges[:-1] + edges[1:])
    Z = np.zeros((centres.size, ncol), dtype=float)
    for En, Ik in zip(E, I):
        w = np.exp(-0.5 * ((centres - En) / sigma_ev) ** 2)
        Z += np.outer(w, Ik)

    vmax = np.percentile(Z, 99.9) if np.any(Z > 0) else 1.0
    pos  = Z[Z > 0]
    vmin = np.percentile(pos, 5) if pos.size else 1e-6
    vmin_eff = max(vmin, vmax/float(scaled_vmin))
    if vmin_eff >= vmax:
        vmin_eff = max(vmax * 0.5, 1e-8)
    norm = mcolors.LogNorm(vmin=vmin_eff, vmax=vmax)
    cmap = _fade_cmap(); cmap.set_bad('black'); cmap.set_under('black')

    # ---- PDOS (cumulative, same energy grid "centres") ----
    pdos_weights = compute_pdos_weights(C, S, method="mulliken")
    projected = project_pdos(pdos_weights, shells)
    atom_idx_shell = np.array([int(sh["atom_idx"]) for sh in shells], dtype=int)
    sym_shell      = np.array([sh["sym"] for sh in shells], dtype=object)
    sym_to_atom_indices = {
        sym: np.unique(atom_idx_shell[sym_shell == sym]) for sym in set(sym_shell)
    }

    curves, labels_p = [], []
    for sym in pdos_atom_list:
        rows = sym_to_atom_indices.get(sym, np.array([], dtype=int))
        if rows.size == 0:
            continue
        w_mo = projected[rows, :].sum(axis=0)
        curves.append(broaden(w_mo, eps_eV, centres, sigma_pdos))
        labels_p.append(sym)

    Ycum, xmax = None, 1.0
    if curves:
        Y = np.column_stack(curves)
        Ycum = np.cumsum(Y, axis=1)
        xmax = float(Ycum.max()) * 1.05 if Ycum.size else 1.0

    ef = (fermi_from_occ(eps_eV, occ, ewin) if midgap is None else float(midgap))

    # ---- Layout: [85% fuzzy] | [12% PDOS] | [3% info column] ----
    fig = plt.figure(figsize=(9.6, 5.4), constrained_layout=False)
    gs = GridSpec(1, 100, figure=fig, wspace=0.05)
    axL   = fig.add_subplot(gs[0, :85])      # left fuzzy
    axR   = fig.add_subplot(gs[0, 85:97], sharey=axL)  # PDOS
    infoS = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 97:], height_ratios=[3, 2], hspace=0.15)
    axLeg = fig.add_subplot(infoS[0])        # legend panel (axes turned off)
    axC   = fig.add_subplot(infoS[1])        # colorbar panel

    # ---- Draw fuzzy (left) ----
    extent = [0, ncol-1, centres.min(), centres.max()]
    im = axL.imshow(Z, origin='lower', aspect='auto', extent=extent, cmap=cmap, norm=norm)
    axL.set_facecolor('black')
    axL.set_ylim(ewin)
    axL.set_ylabel("Energy (eV)")
    axL.set_xlabel("High-Symmetry k-Path")
    # HS ticks
    tick_pos, tick_lab, prev = [], [], None
    for i, lbl in enumerate(labels_dedup):
        if lbl and lbl != prev:
            tick_pos.append(float(i))
            tick_lab.append(_plain_k_label(lbl))
            axL.axvline(i, color='gray', lw=0.5, alpha=0.6)
            prev = lbl
    axL.set_xticks(tick_pos); axL.set_xticklabels(tick_lab)
    axL.axhline(ef, color='w', ls='--', lw=1.2, alpha=0.9)

    # ---- Draw PDOS (right-mid), hide duplicate y ----
    axR.tick_params(left=False, labelleft=False)
    axR.spines["left"].set_visible(False)
    if Ycum is not None:
        prev = np.zeros_like(centres)
        handles = []
        for j in range(Ycum.shape[1]):
            p = axR.fill_betweenx(centres, prev, Ycum[:, j], alpha=0.75, linewidth=0.0, label=labels_p[j])
            handles.append(p)
            prev = Ycum[:, j]
        axR.set_xlim(0.0, xmax)
        axR.set_xlabel("PDOS (a.u.)")
    else:
        handles, labels_p = [], []
        axR.set_xlim(0.0, 1.0); axR.set_xlabel("PDOS (a.u.)")

    # ---- Legend in its own panel (never overlaps) ----
    axLeg.axis("off")
    if labels_p:
        axLeg.legend(handles, labels_p, loc="upper left", frameon=False, fontsize=9)

    # ---- Colorbar below the legend (same narrow info column) ----
    # vertical colorbar aligned under axLeg
    cb = plt.colorbar(im, cax=axC, orientation='vertical')
    cb.set_label("Intensity (arb., log scale)", fontsize=9)

    # tidy figure and save (no tight_layout to avoid warnings with custom axes)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"✓ Combined fuzzy+PDOS plot saved to {outfile}")
    plt.close(fig)

