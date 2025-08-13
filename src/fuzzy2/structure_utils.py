# src/fuzzy2/structure_utils.py

import numpy as np
import seekpath
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from . import config, io_utils

def to_primitive(struct):
    """Returns the primitive standard structure."""
    return SpacegroupAnalyzer(struct).get_primitive_standard_structure()

def setup_dft_structure(lattice_matrix, bulk_xyz_path):
    """Builds and standardizes the pymatgen structure for k-path generation."""
    lat = Lattice(lattice_matrix)
    syms, coords = io_utils.read_xyz(bulk_xyz_path)
    struct = Structure(
        lattice=lat,
        species=syms,
        coords=coords,
        coords_are_cartesian=True,
    )
    sga = SpacegroupAnalyzer(to_primitive(struct))
    struct_conv = sga.get_conventional_standard_structure()
    print(f"[INFO] Built conventional cell with {len(struct_conv)} atoms for k-path generation.")
    return struct_conv

def get_kpoints_cart(struct, line_density=50):
    """
    Robust wrapper for pymatgen's k-path generation, returning Cartesian coordinates.
    """
    kp = HighSymmKpath(struct)
    kpts, labels = kp.get_kpoints(line_density=line_density, coords_are_cartesian=True)
    return np.asarray(kpts), labels

def print_lattice_info(name, lattice):
    """Prints lattice and reciprocal lattice vectors."""
    print(f"\n[INFO] {name} lattice vectors (Å):")
    for i, v in enumerate(lattice):
        print(f"    a{i+1}: [{v[0]:8.4f}  {v[1]:8.4f}  {v[2]:8.4f}]")
    rec = 2 * np.pi * np.linalg.inv(lattice).T
    print(f"[INFO] {name} reciprocal lattice vectors (Å⁻¹):")
    for i, v in enumerate(rec):
        print(f"    b{i+1}: [{v[0]:8.4f}  {v[1]:8.4f}  {v[2]:8.4f}]")

def print_kpath_info(struct, tag):
    """Prints detailed crystal symmetry and k-path information."""
    sga = SpacegroupAnalyzer(struct, symprec=1e-3)
    print(f"\n--- [{tag}] Crystal Symmetry & K-Path ---")
    print(f"  Space group:       {sga.get_space_group_symbol()} (No. {sga.get_space_group_number()})")
    print(f"  Crystal system:    {sga.get_crystal_system()}")
    print(f"  Lattice params (Å): a={struct.lattice.a:.4f}, b={struct.lattice.b:.4f}, c={struct.lattice.c:.4f}")
    print(f"  Lattice angles:    α={struct.lattice.alpha:.2f}, β={struct.lattice.beta:.2f}, γ={struct.lattice.gamma:.2f}")

    kp = HighSymmKpath(struct)
    if kp.kpath is None:
        print("\n[WARNING] No standard high-symmetry k-path found for this structure.")
        return
    print("\n  High-symmetry points (Cartesian Å⁻¹):")
    B = struct.lattice.reciprocal_lattice.matrix
    for lbl, frac in kp.kpath["kpoints"].items():
        cart = np.dot(frac, B)
        print(f"    {lbl:<5} -> [{cart[0]:8.4f}, {cart[1]:8.4f}, {cart[2]:8.4f}]")
    
    print("\n  K-path segments:")
    for seg in kp.kpath['path']:
        print("    " + " → ".join(seg))

def k_path_distance(kpts_cart):
    """Calculates the cumulative distance along a path of k-points."""
    dk = np.linalg.norm(np.diff(kpts_cart, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(dk)))

def build_shell_dicts(syms, coords_ang, basis_dict):
    """Creates a list of shell dictionaries required by libint_fuzzy."""
    shells = []
    for atom_idx, (sym, xyz_ang) in enumerate(zip(syms, coords_ang)):
        if sym not in basis_dict:
            raise KeyError(f"Basis set not found for element '{sym}'. Check your basis file and name.")
        xyz_bohr = np.asarray(xyz_ang, dtype=float) * config.BOHR_PER_ANGSTROM
        for l, exps, coefs in basis_dict[sym]:
            # The C++ extension expects pure=True for spherical harmonics
            shells.append(dict(sym=sym, atom_idx=atom_idx, l=l, exps=exps, coefs=coefs, center=xyz_bohr, pure=True))
    return shells

