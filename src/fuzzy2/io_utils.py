# src/fuzzy2/io_utils.py

import collections
import numpy as np
import os
import re
import time
from scipy.sparse import issparse, csr_matrix, save_npz, load_npz
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

# --- Structure Parsers (from fuzzy2.py) ---

def load_structure(cif_path):
    """Loads a structure from a CIF file and returns the primitive cell."""
    if not cif_path:
        raise ValueError("A CIF path must be provided.")
    return CifParser(cif_path).parse_structures(primitive=True)[0]

def read_xyz(path):
    """Reads atom symbols and coordinates from a standard XYZ file."""
    with open(path) as f:
        lines = f.readlines()
    nat = int(lines[0].strip())
    syms = []
    coords = []
    for l in lines[2:2+nat]:
        p = l.split()
        syms.append(p[0])
        coords.append(tuple(map(float, p[1:4])))
    return syms, np.asarray(coords)

# --- Basis and MO Parsers (from parsers.py) ---

def save_mo_csr_singlefile(C, eps, occ, outpath):
    """Stores C (CSR or dense), eps, occ in a single .npz file."""
    if not issparse(C):
        C = csr_matrix(C)
    np.savez_compressed(
        outpath,
        data=C.data,
        indices=C.indices,
        indptr=C.indptr,
        shape=C.shape,
        eps=eps,
        occ=occ
    )
    print(f"[MOs] Wrote C, eps, occ to: {outpath}")

def read_mos_auto(path, n_ao_total, mmap_path=None, verbose=False):
    from scipy import sparse
    import numpy as np
    import os

    ext = os.path.splitext(path)[-1].lower()
    if ext == ".npz":
        if verbose:
            print(f"[MOs] Detected .npz: {path}")
        d = np.load(path)
        C = sparse.csr_matrix((d['data'], d['indices'], d['indptr']), shape=d['shape'])
        eps = d['eps']
        occ = d['occ']
        if verbose:
            print(f"[MOs] Loaded C shape: {C.shape}, eps: {eps.shape}, occ: {occ.shape}")
        return C, eps, occ
    else:
        # Otherwise parse text, then save .npz for next time
        C, eps, occ = read_mos_txt_streaming(path, n_ao_total, verbose=verbose)
        # Write for future use
        outdir = os.path.dirname(os.path.abspath(path))
        base = os.path.splitext(os.path.basename(path))[0]
        outpath = os.path.join(outdir, base + "_csr.npz")
        save_mo_csr_singlefile(C, eps, occ, outpath)
        return C, eps, occ

def parse_basis(fname, wanted):
    """A robust parser for MOLOPT basis set files."""
    basis = collections.defaultdict(list)
    with open(fname) as f:
        lines = f.readlines()

    line_iter = iter(lines)
    for line in line_iter:
        ln = line.strip()
        if not ln or ln.startswith('#'):
            continue
        parts = ln.split()
        if len(parts) < 2:
            continue
        elem, bname = parts[0], parts[1]
        if bname != wanted:
            # Skip blocks for basis sets we don't want
            try:
                nset = int(next(line_iter).strip().split()[0])
                for _ in range(nset):
                    hdr_line = next(line_iter)
                    nexp = int(hdr_line.split()[3])
                    for _ in range(nexp):
                        next(line_iter)
            except (StopIteration, IndexError, ValueError):
                continue
            continue
        
        # Process the desired basis set block
        try:
            nset = int(next(line_iter).strip().split()[0])
            for _ in range(nset):
                hdr = next(line_iter).split()
                lmin, nexp, counts = int(hdr[1]), int(hdr[3]), list(map(int, hdr[4:]))
                exps_full = []
                coef_rows = []
                for _ in range(nexp):
                    prim_line = next(line_iter).split()
                    exps_full.append(float(prim_line[0]))
                    coef_rows.append([float(c) for c in prim_line[1:]])
                
                coef_cols = np.array(coef_rows).T
                
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
            raise IOError(f"FATAL: Could not parse block for '{elem}' with basis '{bname}'. Details: {e}")
    return basis

_NUM_RE = re.compile(r"""[\+\-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][\+\-]?\d+)?""", re.VERBOSE)

def _extract_numbers(s: str):
    toks = _NUM_RE.findall(s)
    return [float(t.replace('D','E').replace('d','E')) for t in toks]

def _parse_tail_floats(line: str, n_cols: int):
    toks = line.replace('D','E').replace('d','E').strip().rsplit(None, n_cols)
    if not toks: return []
    try:
        if len(toks) > n_cols:
            return np.array(toks[-n_cols:], dtype=float)
        else:
             # This path is more complex, fallback is safer
            raise ValueError
    except ValueError:
        nums = _extract_numbers(line)
        return np.asarray(nums[-n_cols:], dtype=float) if len(nums) >= n_cols else np.asarray(nums, dtype=float)

def read_mos_txt_streaming(path, n_ao_total, *,
                           dtype=np.float32,
                           mmap_path=None,
                           return_memmap=True,
                           verbose=True,
                           debug=False,
                           log_every=200):
    """
    Fast streaming CP2K MO parser with progress and timing info.
    """
    if mmap_path is None:
        mmap_path = os.path.join(os.path.dirname(os.path.abspath(path)), "C_memmap.dat")

    def _is_int_line(s: str) -> bool:
        toks = s.split()
        if not toks: return False
        for t in toks:
            if t.startswith('+'): t = t[1:]
            if not t.isdigit(): return False
        return True

    def _next_nonempty(f):
        for line in f:
            s = line.strip()
            if s:
                return s
        return None

    file_size = os.path.getsize(path) if os.path.exists(path) else 0
    t0 = time.perf_counter()

    # PASS 1: count blocks and widths
    n_mo_total = 0; blocks = []
    with open(path, 'r', buffering=1024*1024) as f:
        while True:
            line = f.readline()
            if not line: break
            s = line.strip()
            if not _is_int_line(s): continue
            n_cols = len(s.split())
            if _next_nonempty(f) is None:
                raise RuntimeError("Unexpected EOF while reading energies line (pass1).")
            if _next_nonempty(f) is None:
                raise RuntimeError("Unexpected EOF while reading occupations line (pass1).")
            ao_count = 0
            while ao_count < n_ao_total:
                coeff_line = f.readline()
                if not coeff_line: raise RuntimeError("Unexpected EOF in coefficients block (pass1).")
                if coeff_line.strip(): ao_count += 1
            blocks.append((n_mo_total, n_cols))
            n_mo_total += n_cols

    if n_mo_total == 0:
        raise RuntimeError("No MO blocks detected in file.")

    order = 'F'
    if return_memmap:
        C = np.memmap(mmap_path, mode='w+', dtype=dtype, shape=(n_ao_total, n_mo_total), order=order)
    else:
        C = np.empty((n_ao_total, n_mo_total), dtype=dtype, order=order)
    eps = np.empty(n_mo_total, dtype=np.float64)
    occ = np.empty(n_mo_total, dtype=np.float64)

    # PASS 2: parse & fill, now with progress reporting
    with open(path, 'r', buffering=1024*1024) as f:
        blk_idx = 0
        n_blocks = len(blocks)
        print(f"[MOs] Starting parse of {n_mo_total} MOs from {os.path.basename(path)}...")
        t1 = time.perf_counter()
        while True:
            line = f.readline()
            if not line: break
            s = line.strip()
            if not _is_int_line(s): continue

            offset, n_cols = blocks[blk_idx]; blk_idx += 1
            col_slice = slice(offset, offset + n_cols)

            # energies (fast path; allow wrapping)
            vals = []
            while len(vals) < n_cols:
                e_line = _next_nonempty(f)
                if e_line is None:
                    raise RuntimeError(f"EOF reading energies at block {blk_idx}.")
                arr = _parse_tail_floats(e_line, n_cols - len(vals))
                if arr.size == 0:
                    nums = _extract_numbers(e_line)
                    arr = np.asarray(nums, dtype=float) if nums else np.array([], float)
                vals.extend(arr.tolist())
            eps[col_slice] = np.asarray(vals[:n_cols], dtype=np.float64)

            # occupations
            vals = []
            while len(vals) < n_cols:
                o_line = _next_nonempty(f)
                if o_line is None:
                    raise RuntimeError(f"EOF reading occupations at block {blk_idx}.")
                arr = _parse_tail_floats(o_line, n_cols - len(vals))
                if arr.size == 0:
                    nums = _extract_numbers(o_line)
                    arr = np.asarray(nums, dtype=float) if nums else np.array([], float)
                vals.extend(arr.tolist())
            occ[col_slice] = np.asarray(vals[:n_cols], dtype=np.float64)

            # coefficients: n_ao_total lines
            ao_row = 0
            while ao_row < n_ao_total:
                coeff_line = f.readline()
                if not coeff_line:
                    raise RuntimeError(f"EOF in coefficients block at block {blk_idx}, ao_row {ao_row}.")
                sline = coeff_line.strip()
                if not sline: continue

                # fast tail (n_cols tokens); if short, we try to wrap
                tail = _parse_tail_floats(sline, n_cols)
                if tail.size < n_cols:
                    acc = tail.tolist()
                    while len(acc) < n_cols:
                        extra = _next_nonempty(f)
                        if extra is None:
                            raise RuntimeError(f"EOF while wrapping coeff line at block {blk_idx}, ao_row {ao_row}.")
                        more = _parse_tail_floats(extra, n_cols - len(acc))
                        if more.size == 0:
                            more = np.asarray(_extract_numbers(extra), dtype=float)
                        acc.extend(more.tolist())
                    tail = np.asarray(acc[-n_cols:], dtype=float)

                C[ao_row, col_slice] = tail.astype(dtype, copy=False)
                ao_row += 1

            # Progress report
            if (blk_idx % log_every == 0) or (blk_idx == 1) or (blk_idx == n_blocks):
                now = time.perf_counter()
                n_done = offset + n_cols
                mb = file_size / (1024 ** 2) if file_size else 0.0
                elapsed = now - t1
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_mo_total - n_done) / rate if rate > 0 else 0
                print(f"[MOs] Block {blk_idx}/{n_blocks} | {n_done}/{n_mo_total} MOs "
                      f"({100*n_done/n_mo_total:.1f}%) | "
                      f"Elapsed: {elapsed:6.1f}s | "
                      f"ETA: {eta:6.1f}s")

    if isinstance(C, np.memmap):
        C.flush()

    t2 = time.perf_counter()
    if verbose:
        mb = file_size / (1024**2) if file_size else 0.0
        print(f"[MOs] Finished parse in {t2-t0:.1f} seconds.")
        print(f"[MOs] Parsed: C shape={C.shape} (dtype={dtype.__name__}, order={order}), eps={eps.shape}, occ={occ.shape})")
        if isinstance(C, np.memmap):
            print(f"[MOs] C memmap: {mmap_path}")
        print(f"[MOs] Input size ~ {mb:,.1f} MB")

    return C, eps, occ



# --- SOC Spinors ---

def load_soc_spinors_npz(npz_path, verbose=True):
    """Load SOC spinors from an .npz file with keys:
    indices, energies_Ha, energies_eV, occupations, U (complex128).
    Returns (U, eps_eV, occ, indices).
    """
    import time, os
    t0 = time.perf_counter()
    data = np.load(npz_path)
    required = ["energies_eV", "U"]
    for k in required:
        if k not in data.files:
            raise KeyError(f"[soc_npz] Missing required key '{k}' in {npz_path}. Found: {data.files}")
    indices = data["indices"] if "indices" in data.files else None
    eps_eV  = data["energies_eV"]
    occ     = data["occupations"] if "occupations" in data.files else None
    U       = data["U"]
    if np.iscomplexobj(U) is False:
        U = U.astype(np.complex128)
    if U.ndim != 2:
        raise ValueError(f"[soc_npz] U must be 2D, got shape {U.shape}")
    if eps_eV.shape[0] != U.shape[1]:
        raise ValueError(f"[soc_npz] energies length {eps_eV.shape[0]} != n_spinors {U.shape[1]}")
    if verbose:
        took = time.perf_counter() - t0
        print(f"[SOC] Loaded spinors: U {U.shape} (complex), energies {eps_eV.shape}, in {took:.2f}s from {os.path.basename(npz_path)}")
    return U, eps_eV, occ, indices

def configure_threading(nthreads: int, blas_threads: int = 1, quiet: bool = False):
    """
    Configure OpenMP (Libint/Eigen) and BLAS threading to avoid oversubscription.
    - OMP: nthreads, bind threads to cores
    - BLAS: blas_threads (default 1 to avoid oversubscription)
    If threadpoolctl is present, adjusts pools at runtime too.
    """
    import os

    nthreads = max(1, int(nthreads))
    if blas_threads is None:
        blas_threads = 1
    blas_threads = int(blas_threads)

    # -- Set env defaults if not already set --
    def _set_default(key, val):
        if key not in os.environ or not os.environ[key].strip():
            os.environ[key] = str(val)

    _set_default("OMP_NUM_THREADS", nthreads)
    _set_default("OMP_PROC_BIND", "close")
    _set_default("OMP_PLACES", "cores")
    if blas_threads > 0:
        _set_default("MKL_NUM_THREADS", blas_threads)
        _set_default("OPENBLAS_NUM_THREADS", blas_threads)
        _set_default("NUMEXPR_NUM_THREADS", blas_threads)

    # -- Try runtime adjustment via threadpoolctl --
    set_rt = False
    info = []
    try:
        from threadpoolctl import threadpool_limits, threadpool_info
        if blas_threads > 0:
            threadpool_limits(blas_threads, user_api=["blas", "openmp", "mkl", "openblas"])
            set_rt = True
        info = threadpool_info()
    except Exception:
        pass

    # -- Summary printout --
    if not quiet:
        def _get(k): return os.environ.get(k, "<unset>")
        print(f"[threads] OMP_NUM_THREADS={_get('OMP_NUM_THREADS')}  "
              f"OMP_PROC_BIND={_get('OMP_PROC_BIND')}  OMP_PLACES={_get('OMP_PLACES')}")
        print(f"[threads] MKL_NUM_THREADS={_get('MKL_NUM_THREADS')}  "
              f"OPENBLAS_NUM_THREADS={_get('OPENBLAS_NUM_THREADS')}  "
              f"(runtime set: {'yes' if set_rt else 'no'})")
        if info:
            rows = [f"    {lib.get('internal_api','?'):>7} | {lib.get('num_threads','?'):>2} | {lib.get('filename','?')}"
                    for lib in info]
            print("[threads] loaded pools:\n" + "\n".join(rows))

