import collections
import numpy as np 

#---- Parser Basis set ---------#
def parse_basis(fname, wanted):
    """
    A robust parser for MOLOPT basis set files that correctly handles the
    file's block structure and general contractions.
    """
    basis = collections.defaultdict(list)
    with open(fname) as f:
        lines = f.readlines()

    line_iter = iter(lines)
    for line in line_iter:
        ln = line.strip()
        if not ln or ln.startswith('#'):
            continue

        parts = ln.split()
        # A valid element header must have at least 2 words
        if len(parts) < 2:
            continue

        elem, bname = parts[0], parts[1]

        # If we find a block for a basis set we don't want, we must
        # figure out how many lines it has and skip over them.
        if bname != wanted:
            try:
                nset_line = next(line_iter)
                nset = int(nset_line.strip().split()[0])
                for _ in range(nset):
                    hdr_line = next(line_iter)
                    nexp = int(hdr_line.split()[3])
                    # Skip the nexp primitive lines
                    for _ in range(nexp):
                        next(line_iter)
            except (StopIteration, IndexError, ValueError):
                # Malformed block or end of file, continue scanning
                continue
            # Successfully skipped, continue to the next potential header line
            continue

        # If we are here, we found the element and basis we want. Process it.
        try:
            nset_line = next(line_iter)
            nset = int(nset_line.strip().split()[0])
            for _ in range(nset):
                hdr_line = next(line_iter)
                hdr = hdr_line.split()
                lmin, nexp = int(hdr[1]), int(hdr[3])
                counts = list(map(int, hdr[4:]))
                n_contractions = sum(counts)

                exps_full = []
                coef_rows = []
                for _ in range(nexp):
                    prim_line = next(line_iter).split()
                    exps_full.append(float(prim_line[0]))
                    coef_rows.append([float(c) for c in prim_line[1:]])
                
                coef_cols = np.array(coef_rows).T
                
                if coef_cols.shape[0] != n_contractions:
                    raise ValueError(f"Basis parse error for {elem}: {n_contractions} contractions expected, but {coef_cols.shape[0]} found.")
                
                coef_idx = 0
                for j, num_shells_for_l in enumerate(counts):
                    l = lmin + j
                    for _ in range(num_shells_for_l):
                        current_coeffs = coef_cols[coef_idx]
                        nonzero_mask = np.abs(current_coeffs) > 1e-12
                        
                        final_exps = np.asarray(exps_full)[nonzero_mask]
                        final_coefs = current_coeffs[nonzero_mask]
                        
                        basis[elem].append((l, final_exps, final_coefs))
                        coef_idx += 1
        except (StopIteration, IndexError, ValueError) as e:
            raise IOError(f"FATAL: Could not parse block for '{elem}' with basis '{bname}'. File may be malformed. Details: {e}")

    return basis

# ----- Parse Molecular Orbitals -----
def read_mos_txt(path, n_ao_total):
    """
    A robust parser for CP2K MO output files, designed to handle
    molecular systems with multiple atoms and basis functions.
    It takes the total number of AOs as an argument and correctly
    handles blank lines within coefficient blocks.
    """
    with open(path) as f:
        lines = f.readlines()

    all_eps = []
    all_occ = []
    all_c_cols = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        tokens = line.split()
        
        # Check for a header line (e.g., "1 2" or just "3")
        if len(tokens) > 0 and all(t.isdigit() for t in tokens):
            n_cols = len(tokens)
            
            # The next non-empty line has energies
            i += 1
            if i >= len(lines): break
            eps = [float(e) for e in lines[i].strip().split()]
            all_eps.extend(eps)

            # The next non-empty line has occupations
            i += 1
            if i >= len(lines): break
            occ = [float(o) for o in lines[i].strip().split()]
            all_occ.extend(occ)
            i += 1

            # Read the coefficient block, skipping blank lines
            c_block = np.zeros((n_ao_total, n_cols))
            ao_lines_read = 0
            while ao_lines_read < n_ao_total and i < len(lines):
                coeff_line = lines[i].strip()
                i += 1 # Consume the line
                
                if not coeff_line: # Skip blank lines between atom blocks
                    continue
                
                parts = coeff_line.split()
                try:
                    # The last n_cols are the coefficients
                    coeffs = [float(c) for c in parts[-n_cols:]]
                    if len(coeffs) != n_cols:
                        raise ValueError(f"Incorrect number of coefficients on line {i}")
                    c_block[ao_lines_read, :] = coeffs
                    ao_lines_read += 1
                except (ValueError, IndexError) as e:
                     raise ValueError(f"Error parsing coefficient line {i}: '{coeff_line}'\n{e}")
            
            if ao_lines_read != n_ao_total:
                raise RuntimeError(f"Expected to read {n_ao_total} AO lines, but only found {ao_lines_read} for MOs {tokens}")

            all_c_cols.append(c_block)
        else:
            i += 1
            
    if not all_c_cols:
        raise RuntimeError("Could not parse any valid MO coefficient blocks from the file.")

    C = np.hstack(all_c_cols)
    eps_arr = np.asarray(all_eps)
    occ_arr = np.asarray(all_occ)
    
    print(f"Successfully parsed MOs. Final C matrix shape: {C.shape}")
    return C, eps_arr, occ_arr

import numpy as np
import itertools

# ----- Parse Molecular Orbitals (Optimized) -----
def read_mos_txt2(path, n_ao_total):
    """
    An efficient parser for CP2K MO output files.

    It uses a two-pass approach to pre-allocate memory, minimizing
    memory usage and improving speed.
    """
    # 1. First Pass: Pre-scan to get total number of MOs
    n_mo_total = 0
    with open(path) as f:
        for line in f:
            tokens = line.strip().split()
            # Check for a header line (e.g., "1 2" or just "3")
            if len(tokens) > 0 and all(t.isdigit() for t in tokens):
                n_mo_total += len(tokens)

    if n_mo_total == 0:
        raise RuntimeError("Could not find any MOs in the file.")

    # 2. Pre-allocate NumPy arrays
    C = np.zeros((n_ao_total, n_mo_total))
    eps_arr = np.zeros(n_mo_total)
    occ_arr = np.zeros(n_mo_total)
    
    # 3. Second Pass: Parse and fill the arrays
    mo_col_idx = 0 # Current column index in the final matrices
    with open(path) as f:
        # Use an iterator for more control
        f_iter = iter(f)
        for line in f_iter:
            tokens = line.strip().split()
            
            if len(tokens) > 0 and all(t.isdigit() for t in tokens):
                n_cols_in_block = len(tokens)
                
                # Define the slice for placing data from this block
                block_slice = slice(mo_col_idx, mo_col_idx + n_cols_in_block)

                # Read energies and occupations
                eps_line = next(f_iter).strip().split()
                occ_line = next(f_iter).strip().split()
                eps_arr[block_slice] = [float(e) for e in eps_line]
                occ_arr[block_slice] = [float(o) for o in occ_line]

                # Read the coefficient block
                ao_row_idx = 0
                while ao_row_idx < n_ao_total:
                    coeff_line = next(f_iter).strip()
                    if not coeff_line:  # Skip blank lines
                        continue
                    
                    parts = coeff_line.split()
                    coeffs = [float(c) for c in parts[-n_cols_in_block:]]
                    C[ao_row_idx, block_slice] = coeffs
                    ao_row_idx += 1

                mo_col_idx += n_cols_in_block
    
    print(f"Successfully parsed MOs. Final C matrix shape: {C.shape}")
    return C, eps_arr, occ_arr
