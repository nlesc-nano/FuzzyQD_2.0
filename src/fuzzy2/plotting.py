# src/fuzzy2/plotting.py

import numpy as np
import re 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter
from . import config

def _fmt_k_label(lbl: str) -> str:
    s = str(lbl).strip()
    # If label already has TeX math delimiters, trust it as-is.
    if "$" in s:
        return s
    # Otherwise wrap in bold math
    return rf"$\mathbf{{{s}}}$"

def _plain_k_label(lbl: str) -> str:
    s = str(lbl).strip()
    # Remove math $...$
    s = s.replace("$", "")
    # Unwrap common TeX wrappers
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"[{}]", "", s)
    # Map TeX names to Unicode where helpful
    s = s.replace(r"\Gamma", "Γ")
    s = s.replace(r"\Delta", "Δ")
    s = s.replace(r"\Sigma", "Σ")
    return s

def write_kpath_to_file(kpts_cart, labels, k_path_dist, outfile="kpath_data.txt"):
    """
    Saves the detailed k-path information to a text file for debugging.

    Args:
        kpts_cart (np.ndarray): Cartesian coordinates of k-points (N_k, 3).
        labels (list): High-symmetry point labels for each k-point.
        k_path_dist (np.ndarray): Cumulative distance along the k-path (N_k,).
        outfile (str): Name of the output text file.
    """
    header = (
        f"{'Index':>6s} {'k_x':>12s} {'k_y':>12s} {'k_z':>12s} "
        f"{'Distance':>14s}   {'Label':<5s}\n"
        f"{'='*6} {'='*12} {'='*12} {'='*12} {'='*14}   {'='*5}\n"
    )
    with open(outfile, "w") as f:
        f.write(header)
        for i, (k, d, lbl) in enumerate(zip(kpts_cart, k_path_dist, labels)):
            label_str = lbl if lbl else "-"
            f.write(
                f"{i:>6d} {k[0]:>12.6f} {k[1]:>12.6f} {k[2]:>12.6f} "
                f"{d:>14.6f}   {label_str:<5s}\n"
            )
    print(f"  ✓ Detailed k-path data saved to: {outfile}")

def _fade_cmap():
    """Creates a black-to-inferno colormap."""
    base = plt.colormaps.get_cmap("inferno")
    cols = [(0, 0, 0)] + list(base(np.linspace(0, 1, 256)))
    return mcolors.LinearSegmentedColormap.from_list("inf_bw", cols)

def dedup_kpath_strict(kpts, labels, tol=1e-8):
    """
    Deduplicate k-points only if both coordinate and label are identical.
    This preserves distinct high-symmetry points at segment boundaries.
    """
    keep = [0]
    for i in range(1, len(kpts)):
        same_coord = np.linalg.norm(kpts[i] - kpts[keep[-1]]) < tol
        same_label = labels[i] == labels[keep[-1]]
        if not (same_coord and same_label):
            keep.append(i)
    return kpts[keep], [labels[i] for i in keep], np.array(keep)

def fuzzy_band_imshow(kpts_cart, labels, eps_Ha, intensity,
                      dE=0.01, energy_window=(-5, 5), blur_sigma=None, outfile=None):
    """
    Generates and saves the main fuzzy band structure plot.
    """
    kpts_dedup, labels_dedup, keep_indices = dedup_kpath_strict(np.asarray(kpts_cart), labels)
    intensity_dedup = intensity[:, keep_indices]
    
    # 1. Energy Grid
    E_eV = eps_Ha * config.HARTREE_TO_EV
    mask = (E_eV >= energy_window[0]) & (E_eV <= energy_window[1])
    E_use, I_use = E_eV[mask], intensity_dedup[mask, :]
    if E_use.size == 0:
        print("[WARNING] No MOs found in the specified energy window. Skipping plot.")
        return
        
    e_min, e_max = energy_window
    e_grid = np.arange(e_min, e_max, dE)
    
    # 2. Project intensity onto grid
    Z = np.zeros((len(e_grid), I_use.shape[1]))
    sigma = 0.05  # Gaussian broadening in eV
    for E_n, psi_k_sq in zip(E_use, I_use):
        w = np.exp(-0.5 * ((e_grid - E_n) / sigma)**2)
        Z += np.outer(w, psi_k_sq)
        
    if blur_sigma:
        Z = gaussian_filter(Z, sigma=blur_sigma)

    # 3. Plotting Setup
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    x_grid = np.arange(Z.shape[1])
    extent = [x_grid.min(), x_grid.max(), e_grid.min(), e_grid.max()]
    
    vmax = np.percentile(Z, 99.8)
    vmin = max(np.percentile(Z[Z > 1e-9], 5), vmax / 1e4) if np.any(Z > 1e-9) else vmax / 1e4
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    
    im = ax.imshow(Z, origin='lower', aspect='auto', extent=extent, cmap=_fade_cmap(), norm=norm)
    
    # 4. Ticks and Labels
    tick_pos, tick_lab, prev = [], [], None
    for x, lbl in zip(kx, labels_dedup):
        if lbl != prev:
            tick_pos.append(x)
            tick_lab.append(_plain_k_label(lbl))
            ax.axvline(x, color='gray', lw=0.5, alpha=0.6)
            prev = lbl
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    
    for xc in tick_pos:
        ax.axvline(xc, color='w', lw=0.4, alpha=0.6)
    
    ax.set_ylim(energy_window)
    ax.set_xlabel("High-Symmetry k-Path")
    ax.set_ylabel("Energy (eV)")
    ax.set_facecolor('black')
    
    cb = plt.colorbar(im, ax=ax, pad=0.02, extend='max')
    cb.set_label(r"$|\Psi(\mathbf{k})|^2$ (a.u., log scale)", fontsize=11)
    
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
        print(f"✓ Plot saved to {outfile}")
    else:
        plt.show()

def plot_mo_intensity(kpts_cart, labels, k_path_dist, intensity, eps_Ha, mo_indices, gamma_norm=None, outfile_prefix="mo_intensity"):
    """
    Plots the projection intensity |Ψ_n(k)|² for selected MOs.
    """
    kpts_dedup, labels_dedup, keep_indices = dedup_kpath_strict(np.asarray(kpts_cart), labels)
    intensity_dedup = intensity[:, keep_indices]
    k_path_dedup = k_path_dist[keep_indices]

    for n in mo_indices:
        I_k = intensity_dedup[n]
        if gamma_norm is not None:
            I_k = I_k**gamma_norm
        E_eV = eps_Ha[n] * config.HARTREE_TO_EV
        
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(k_path_dedup, I_k, lw=1.0)
        ax.set_title(f"MO {n+1}   $E = {E_eV:.3f}$ eV")
        ax.set_ylabel(r"$|\Psi(\mathbf{k})|^2$" + (f"$^{{{gamma_norm}}}$" if gamma_norm else ""))
        ax.set_xlabel("k-Path Distance (Å⁻¹)")
        ax.set_xlim(k_path_dedup.min(), k_path_dedup.max())
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
        tick_pos, tick_lab, prev_lbl = [], [], None
        for x, lbl in zip(k_path_dedup, labels_dedup):
            if lbl and lbl != prev_lbl:
                tick_pos.append(x)
                tick_lab.append(_plain_k_label(lbl))
                ax.axvline(x, color='gray', lw=0.5, alpha=0.7)
                prev_lbl = lbl
        
        secax = ax.secondary_xaxis('top')
        secax.set_xticks(tick_pos)
        secax.set_xticklabels(tick_lab)

        plt.tight_layout()
        outfile = f"{outfile_prefix}_{n+1}.png"
        plt.savefig(outfile, dpi=200)
        print(f"✓ MO intensity plot saved to {outfile}")
        plt.close(fig)

def plot_fuzzy_map_spinors(kpts_cart, labels, k_path_dist, energies_eV, intensity,
                           ewin, sigma_ev=0.10, gamma_norm=None, scaled_vmin=1e-4, 
                           outfile="fuzzy_soc.png", blur_sigma=None):
    # 0) deduplicate like your original (keep HS endpoints)
    kpts_cart = np.asarray(kpts_cart)
    from .plotting import dedup_kpath_strict as _dedup   # if defined elsewhere
    kpts_dedup, labels_dedup, keep = _dedup(kpts_cart, labels)
    I  = intensity[:, keep]            # (n_states, n_k')
    ncol = I.shape[1]
#    use_true_distance = False
#    if use_true_distance:
#        kdist = np.asarray(k_path_dist, dtype=float)    # cumulative arc-length
#        kdist = kdist[keep] 
#    else:
#        tick_pos = [i for i, lbl in enumerate(labels_dedup) if lbl]  # crude but works

    # 1) uniform pixel grid: one column per k-point
    x_min, x_max = 0.0, float(ncol - 1)

    # 2) select energies in window + gaussian broaden along E
    E = np.asarray(energies_eV)
    in_win = (E >= ewin[0] - 4*sigma_ev) & (E <= ewin[1] + 4*sigma_ev)
    E = E[in_win]; I = I[in_win, :]

    dE = max(0.5*sigma_ev, 0.01)                 # energy bin step (eV)
    edges   = np.arange(ewin[0], ewin[1] + dE, dE)
    centres = 0.5 * (edges[:-1] + edges[1:])
    Z = np.zeros((centres.size, ncol), dtype=float)

    # accumulate gaussian weights
    for En, Ik in zip(E, I):                      # Ik shape: (n_k',)
        w = np.exp(-0.5 * ((centres - En) / sigma_ev) ** 2)
        Z += np.outer(w, Ik)
    if gamma_norm is not None:
        Z = np.power(np.maximum(Z, 0.0), gamma_norm)

    # 3) color normalization (robust, log)
    vmax = np.percentile(Z, 99.9)
    pos  = Z[Z > 0]
    vmin = np.percentile(pos, 5) if pos.size else 1e-6
    scaled_vmax = vmax/scaled_vmin
    norm = mcolors.LogNorm(vmin=max(vmin, scaled_vmax), vmax=vmax)

    # 4) ticks at segment boundaries using labels (plain text, no TeX)
    tick_pos, tick_lab, prev = [], [], None
    for i, lbl in enumerate(labels_dedup):
        if lbl and lbl != prev:
            tick_pos.append(float(i)) 
            tick_lab.append(_plain_k_label(lbl))
            prev = lbl

    # 5) draw
    fig, ax = plt.subplots(figsize=(9,5), facecolor="white")
    extent = [x_min, x_max, centres.min(), centres.max()]
    im = ax.imshow(Z, origin='lower', aspect='auto', extent=extent,
                   cmap=plt.colormaps.get_cmap("inferno"), norm=norm)
    ax.set_ylim(ewin)
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, fontsize=11)
    for xc in tick_pos:
        ax.axvline(xc, color='gray', lw=0.5, alpha=0.6)
    ax.set_xlabel("k-path distance (Å⁻¹)")
    ax.set_ylabel("Energy (eV)")
    cb = plt.colorbar(im, ax=ax, pad=0.02, extend='both')
    cb.set_label("Intensity (arb.)", fontsize=11)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300); print("plot →", outfile)
    plt.close(fig)

