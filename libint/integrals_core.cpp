#include "integrals_core.hpp"
#include <omp.h>
#include <cmath> // For std::tgamma and std::pow
#include <complex>   // <-- added
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace licpp {

/* =================== Parallel launcher ========================== */
template <typename Lambda>
inline void parallel_do(Lambda&& body, int nthreads) {
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
  {
    body(omp_get_thread_num());
  }
#else
  std::vector<std::thread> workers;
  for (int id = 1; id < nthreads; ++id)
    workers.emplace_back(body, id);
  body(0);
  for (auto& t : workers) t.join();
#endif
}

/* ---------------- basis helpers ------------------------------------- */
size_t nbasis(const std::vector<libint2::Shell>& shells) {
  size_t n = 0;
  for (auto& s : shells) n += s.size();
  return n;
}
size_t max_nprim(const std::vector<libint2::Shell>& shells) {
  size_t m = 0;
  for (auto& s : shells) m = std::max(m, s.nprim());
  return m;
}
int max_l(const std::vector<libint2::Shell>& shells) {
  int m = 0;
  for (auto& s : shells)
    for (auto& c : s.contr) m = std::max(m, c.l);
  return m;
}
std::vector<size_t>
map_shell_to_basis_function(const std::vector<libint2::Shell>& shells) {
  std::vector<size_t> s2bf;
  s2bf.reserve(shells.size());
  size_t acc = 0;
  for (auto& s : shells) {
    s2bf.push_back(acc);
    acc += s.size();
  }
  return s2bf;
}

/* -------- generic engine farm with explicit thread count ------------ */
template <libint2::Operator OP, typename OpParams = typename
        libint2::operator_traits<OP>::oper_params_type>
std::vector<Matrix> compute_multipoles(
        const std::vector<libint2::Shell>& shells,
        int nthreads,
        OpParams params = OpParams{}) {

  constexpr unsigned int NP = libint2::operator_traits<OP>::nopers;
  const auto nbf = nbasis(shells);
  std::vector<Matrix> result(NP, Matrix::Zero(nbf, nbf));

  /* one engine per thread */
  std::vector<libint2::Engine> engines(nthreads);
  engines[0] = libint2::Engine(OP, max_nprim(shells), max_l(shells), 0);
  engines[0].set_params(params);
  for (int i = 1; i < nthreads; ++i) engines[i] = engines[0];

  const auto s2bf = map_shell_to_basis_function(shells);
  const int nS = static_cast<int>(shells.size());

  auto worker = [&](int tid) {
    const auto& buf = engines[tid].results();
    for (int s1 = 0; s1 < nS; ++s1) {
      int bf1 = s2bf[s1], n1 = shells[s1].size();
      for (int s2 = 0; s2 <= s1; ++s2) {
        if ( (s2 + s1*nS) % nthreads != tid) continue;
        int bf2 = s2bf[s2], n2 = shells[s2].size();

        engines[tid].compute(shells[s1], shells[s2]);

        // ======================= YOUR NEW DEBUG BLOCK =======================
//        const auto& s1_info = shells[s1];
//        const auto& s2_info = shells[s2];
//
        // Check if we have a (p,d) or (d,p) pair
//        bool is_pd_pair = (s1_info.contr[0].l == 1 && s2_info.contr[0].l == 2) || 
//                          (s1_info.contr[0].l == 2 && s2_info.contr[0].l == 1);

//        if (is_pd_pair) {
//            int buf_n1 = s1_info.size();
//            int buf_n2 = s2_info.size();
            
//            std::cout << "\n[SOC-LIBINT DEBUG]: Raw buffer for shell pair (" << s1 << ", " << s2 
//                      << ") with L=(" << s1_info.contr[0].l << "," << s2_info.contr[0].l << ")"
//                      << " size " << buf_n1 << "x" << buf_n2 << std::endl;

            // Print the raw buffer from Libint
//            for (int i = 0; i < buf_n1 * buf_n2; ++i) {
//                std::cout << buf[0][i] << " ";
//            }
//            std::cout << std::endl;
//        }
        // ====================================================================


        for (unsigned op = 0; op < NP; ++op) {
//          Eigen::Map<const Matrix> block(buf[op], n1, n2);
          Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> block(buf[op], n1, n2);
          result[op].block(bf1,bf2,n1,n2) = block;
          if (s1 != s2)
            result[op].block(bf2,bf1,n2,n1) = block.transpose();
        }
      }
    }
  };
  parallel_do(worker, nthreads);
  return result;
}

/* ---------------- user-facing wrappers ------------------------------ */
Matrix overlap(const std::vector<libint2::Shell>& shells, int nthreads)
{
  return compute_multipoles<libint2::Operator::overlap>(shells, nthreads)[0];
}

Matrix cross_overlap(const std::vector<libint2::Shell>& shells,
                     size_t n_ao, size_t n_prj, int nthreads)
{
  Matrix S = overlap(shells, nthreads);           // AO+proj × AO+proj
  return S.block(0, n_ao, n_ao, n_prj);           // top‑right block
}

std::vector<Matrix> dipole(const std::vector<libint2::Shell>& shells,
                           const std::array<double,3>& origin,
                           int nthreads)
{
  return compute_multipoles<libint2::Operator::emultipole1>(shells, nthreads, origin);
}

/* =======================================================================
 *
 * NEW FUNCTION TO COMPUTE HGH PROJECTOR OVERLAPS VIA DERIVATIVES
 *
 * ======================================================================= */

double hgh_norm_prefactor(int l, int i, double r_l) {
    if (i < 1) throw std::runtime_error("Projector index 'i' must be 1‑based.");
    /* The √2 is *not* part of the original GTH/HGH normalisation.      */
    /* (Goedecker et al., PRB 54 (1996) 1703, Eq. (6) and Table I.)     */
    double num = 1.0;                                   //  <── remove √2
    double gamma_arg = l + (4.0 * i - 1.0) * 0.5;
    double rl_pow    = std::pow(r_l, gamma_arg);
    double den       = rl_pow * std::sqrt(std::tgamma(gamma_arg));
    return num / den;
}

// Helper to compute the normalization constant for a single solid-harmonic GTO primitive
// This is needed to undo the automatic normalization applied by Libint.
double libint_primitive_norm(int l, double alpha) {
    double res = std::pow(2.0 * alpha / M_PI, 0.75);
    double num = std::pow(4.0 * alpha, static_cast<double>(l) / 2.0);
    double den = 0.0;
    if (l > 0) {
        double dfact = 1.0;
        for (int i = 2*l-1; i > 0; i -= 2) {
            dfact *= i;
        }
        den = std::sqrt(dfact);
    } else {
        den = 1.0;
    }
    return res * (num / den);
}

Matrix compute_hgh_projector_overlaps(
    const std::vector<libint2::Shell>& ao_shells,
    const std::vector<HghProjectorParams>& projectors,
    int nthreads)
{
    const auto n_ao = nbasis(ao_shells);
    size_t n_proj_funcs = 0;
    for (const auto& p : projectors)
        n_proj_funcs += (2 * p.l + 1);
    Matrix B = Matrix::Zero(n_ao, n_proj_funcs);
    const auto ao_s2bf = map_shell_to_basis_function(ao_shells);

    int max_l_val = max_l(ao_shells);
    for (const auto& p : projectors)
        max_l_val = std::max(max_l_val, p.l);
    size_t max_nprim_val = max_nprim(ao_shells);

    libint2::Engine engine(libint2::Operator::overlap, max_nprim_val, max_l_val, 0);
    const auto& buf = engine.results();

    size_t proj_col_offset = 0;
    size_t proj_no = 0;
    for (const auto& p : projectors) {
        const int l = p.l;
        const int i = p.i;
        const double r_l = p.r_l;
        const int n_funcs = 2 * l + 1;
        const int k = i - 1;
        const double alpha = 0.5 / (r_l * r_l);

        // Build raw overlap block (finite-difference for k>0)
        Matrix block = Matrix::Zero(n_ao, n_funcs);
        auto compute_block = [&](double a)->Matrix {
            libint2::Shell proj_shell{{a}, {{l, true, {1.0}}}, p.center};

            // ======================= YOUR PRINT STATEMENTS =======================
            // Add these lines to print the information for the projector shell.
//            std::cout << "Projector Shell #" << proj_no 
//                      << ", L: " << l << std::endl;
            
            // Print the exponent 'a' for this shell
//            std::cout << "  Exponent: " << a << std::endl;
            
            // The coefficient for these uncontracted projectors is always {1.0}
//            std::cout << "  Coefficient: " << 1.0 << std::endl;
//            std::cout << "------------------------------------" << std::endl;
            // =====================================================================


            Matrix tmp(n_ao, n_funcs); tmp.setZero();
            for (size_t s1 = 0; s1 < ao_shells.size(); ++s1) {
                engine.compute(ao_shells[s1], proj_shell);
                Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                    computed_block(buf[0], ao_shells[s1].size(), n_funcs);
                tmp.block(ao_s2bf[s1], 0, ao_shells[s1].size(), n_funcs) = computed_block;
            }
            return tmp;
        };

        if      (k == 0) block = compute_block(alpha);
        else if (k == 1) {
            const double delta = 1e-6;
            block = - (compute_block(alpha + delta) - compute_block(alpha - delta)) / (2.0 * delta);
        }
        else if (k == 2) {
            const double delta = 1e-4;
            block = (compute_block(alpha + delta)
                   - 2.0 * compute_block(alpha)
                   + compute_block(alpha - delta)) / (delta * delta);
        }
        else
            throw std::runtime_error("HGH projector: k > 2 not implemented");

        // Compute normalization factor for this projector
        double N_hgh = hgh_norm_prefactor(l, i, r_l);
        double N_gto = libint_primitive_norm(l, alpha);
        double norm_fac = N_hgh / N_gto;

        // Debug: print before/after normalization for each projector
        double max_before = block.cwiseAbs().maxCoeff();
        double max_after;
        // Multiply normalization
        block *= norm_fac;
        max_after = block.cwiseAbs().maxCoeff();

        //printf("[HGH DEBUG] proj#%zu (l=%d,i=%d, alpha=%.5g, r_l=%.5g): N_hgh=%.5g N_gto=%.5g norm_fac=%.5g max|block| before=%.6g after=%.6g\n",
        //    proj_no, l, i, alpha, r_l, N_hgh, N_gto, norm_fac, max_before, max_after);

        // Store result
        B.block(0, proj_col_offset, n_ao, n_funcs) = block;
        proj_col_offset += n_funcs;
        ++proj_no;
    }
    return B;
}

/* -----------------------------------------------------------------------
 *  Real spherical harmonics Y_l^m  in Libint SO order, up to l = 3.
 *  Inputs:  unit direction (x,y,z).  Output array length:
 *     l=0 → 1,  l=1 → 3,  l=2 → 5,  l=3 → 7+3 = 10
 *  Normalisations match Libint’s pure=SO convention (overall constants
 *  absorbed in radial prefactor; relative signs/√ factors are correct for
 *  interference).  Extend similarly for g (l=4) if ever needed.
 * --------------------------------------------------------------------- */
inline void rsh_array_l0(double, double, double, double* o) { o[0] = 1.0; }

inline void rsh_array_l1(double x,double y,double z,double* o) {
  o[0]=y;  o[1]=z;  o[2]=x;
}

inline void rsh_array_l2(double x,double y,double z,double* o) {
  const double r2 = 1.0;
  o[0]=x*y;             // m=-2
  o[1]=y*z;             // m=-1
  o[2]=3.0*z*z-r2;      // m= 0
  o[3]=z*x;             // m=+1
  o[4]=x*x-y*y;         // m=+2
}

inline void rsh_array_l3(double x,double y,double z,double* o) {
  /* SO order for l=3 : m = -3,-2,-1,0,+1,+2,+3  (7 funcs)
     Libint additionally stores the real tesseral family in
     alphabetical xy(y^2-3x^2) …  We mirror CP2K/Libint layout:  */
  o[0] = y*(3.0*x*x - y*y);                // m = -3
  o[1] = 2.0*x*y*z;                         // m = -2
  o[2] = y*(5.0*z*z - 1.0);                 // m = -1   (since r^2=1 on unit sphere)
  o[3] = z*(5.0*z*z - 3.0);                 // m =  0
  o[4] = x*(5.0*z*z - 1.0);                 // m = +1
  o[5] = z*(x*x - y*y);                     // m = +2
  o[6] = x*(x*x - 3.0*y*y);                 // m = +3

}

inline void rsh_array(int l,double x,double y,double z,double* o){
  switch(l){
    case 0: rsh_array_l0(x,y,z,o); break;
    case 1: rsh_array_l1(x,y,z,o); break;
    case 2: rsh_array_l2(x,y,z,o); break;
    case 3: rsh_array_l3(x,y,z,o); break;
    default:
      throw std::runtime_error("rsh_array: l>3 not implemented");
  }
}

/* ---- analytic FT of one Libint shell at a single k-point --------------- */
static std::vector<std::complex<double>>
shell_ft_complex(const libint2::Shell& sh,
                 const Eigen::Vector3d& k) {

  const double klen = k.norm();
  const int l = sh.contr[0].l;
  const size_t nfunc = sh.size();
  std::vector<std::complex<double>> vals(nfunc, std::complex<double>(0.0, 0.0));

  // (-i)^l overall phase for momentum-space solid harmonics
  const std::complex<double> il = std::pow(std::complex<double>(0.0, -1.0), l);

  if (klen < 1e-15) {
    // k -> 0 (Gamma). For l>0, k^l → 0 so the specific direction is irrelevant.
    double ang[9];                     // sufficient up to l=3 (we only use first nfunc)
    rsh_array(l, 0.0, 0.0, 1.0, ang);  // any unit axis

    for (size_t p = 0; p < sh.nprim(); ++p) {
      const double alpha = sh.alpha[p];
      const double coeff = sh.contr[0].coeff[p];
      const double Nl    = 1.0; // // avoid double normalization; matches k>0 branch

      const std::complex<double> pref =
          il * (coeff * Nl)
             * std::pow(M_PI/alpha, 1.5)
             * std::pow(klen/(2.0*alpha), l)
             * std::exp(-klen*klen/(4.0*alpha));

      for (size_t f = 0; f < nfunc; ++f)
        vals[f] += pref * ang[f];
    }
    // phase = 1 at k=0
    return vals;
  }

  // Directional part (real tesseral spherical harmonics in SO order)
  const double x = k(0)/klen, y = k(1)/klen, z = k(2)/klen;
  double ang[9]; rsh_array(l, x, y, z, ang);

  // Radial factor per primitive + normalization and (-i)^l
  for (size_t p = 0; p < sh.nprim(); ++p) {
    const double alpha = sh.alpha[p];
    const double coeff = sh.contr[0].coeff[p];
    const double Nl    = 1.0;  
  // libint_primitive_norm(l, alpha);

    const std::complex<double> pref =
        il * (coeff * Nl)
           * std::pow(M_PI/alpha, 1.5)
           * std::pow(klen/(2.0*alpha), l)
           * std::exp(-klen*klen/(4.0*alpha));

    for (size_t f = 0; f < nfunc; ++f)
      vals[f] += pref * ang[f];
  }

  // center phase e^{-i k·R}
  const Eigen::Vector3d R{sh.O[0], sh.O[1], sh.O[2]};
  const std::complex<double> phase = std::exp(std::complex<double>(0.0, -k.dot(R)));
  for (auto& v : vals) v *= phase;

  return vals;
}

/* ---- real wrapper to keep old code paths working ------------------------ */
static std::vector<double>
shell_ft_real(const libint2::Shell& sh, const Eigen::Vector3d& k) {
  const auto vc = shell_ft_complex(sh, k);
  std::vector<double> vr(vc.size());
  for (size_t i = 0; i < vc.size(); ++i) vr[i] = vc[i].real();
  return vr;
}

/* AO-FT matrix: (n_ao × n_k) real-valued (imag dropped) ------------------ */
Matrix ao_ft(const std::vector<libint2::Shell>& shells,
             const std::vector<KPoint>& kpts,
             int nthreads) {

  const size_t n_ao = nbasis(shells);
  const size_t n_k  = kpts.size();
  Matrix F(n_ao, n_k); F.setZero();
  const auto s2bf = map_shell_to_basis_function(shells);

  parallel_do([&](int tid){
    for (size_t kp = 0; kp < n_k; ++kp) {
      if (kp % nthreads != (size_t)tid) continue;
      Eigen::Vector3d kvec = kpts[kp].vec();
      size_t ao0 = 0;
      for (size_t s = 0; s < shells.size(); ++s) {
        auto vec = shell_ft_real(shells[s], kvec);
        for (size_t f = 0; f < vec.size(); ++f)
          F(ao0 + f, kp) = vec[f];
        ao0 += vec.size();
      }
    }
  }, nthreads);

  return F;
}

/* AO-FT matrix: (n_ao × n_k) complex-valued */
Eigen::MatrixXcd ao_ft_complex(const std::vector<libint2::Shell>& shells,
                               const std::vector<KPoint>& kpts,
                               int nthreads) {
  const size_t n_ao = nbasis(shells);
  const size_t n_k  = kpts.size();
  Eigen::MatrixXcd F(n_ao, n_k); F.setZero();

  parallel_do([&](int tid){
    for (size_t kp = 0; kp < n_k; ++kp) {
      if (kp % nthreads != (size_t)tid) continue;
      Eigen::Vector3d kvec = kpts[kp].vec();
      size_t ao0 = 0;
      for (size_t s = 0; s < shells.size(); ++s) {
        auto vec = shell_ft_complex(shells[s], kvec);
        for (size_t f = 0; f < vec.size(); ++f)
          F(ao0 + f, kp) = vec[f];
        ao0 += vec.size();
      }
    }
  }, nthreads);
  return F;

}


/* ---- Solid-harmonic AO values at a single real-space point --------------
   Evaluates one Libint 'pure' (spherical) shell at r, returning (2l+1) values
   in SO order (same ordering you use for FT and everywhere else).
--------------------------------------------------------------------------- */
static std::vector<double>
shell_values_realspace(const libint2::Shell& sh, const Eigen::Vector3d& r) {
  const int l = sh.contr[0].l;
  const size_t nfunc = sh.size();         // 2l+1 for pure
  std::vector<double> out(nfunc, 0.0);

  const Eigen::Vector3d R = r - Eigen::Vector3d(sh.O[0], sh.O[1], sh.O[2]);
  const double r2 = R.squaredNorm();
  const double rr = std::sqrt(r2);

  // Unit direction; for rr=0, set direction arbitrary, but r^l makes l>0 vanish.
  double nx=0.0, ny=0.0, nz=1.0;
  if (rr > 1e-20) { nx = R[0]/rr; ny = R[1]/rr; nz = R[2]/rr; }

  // Angular array in Libint SO order
  double ang[16]; // enough up to l=3; you already guard l>3 in rsh_array
  rsh_array(l, nx, ny, nz, ang);  // reuses your existing real harmonics
  // Radial factor common to all m: sum_p c_p * N_l(alpha_p) * r^l * exp(-alpha_p r^2)
  double radial = 0.0;
  for (size_t p = 0; p < sh.nprim(); ++p) {
    const double a  = sh.alpha[p];
    const double c  = sh.contr[0].coeff[p];
    const double Nl = 1.0; //libint_primitive_norm(l, a);  // // primitives already normalized by convention used in coeffs
    const double rl = (l==0 ? 1.0 : std::pow(rr, l));
    radial += c * Nl * rl * std::exp(-a * r2);
  }

  for (size_t m = 0; m < nfunc; ++m)
    out[m] = radial * ang[m];

  return out;
}

/* ---- Concatenate AO values for all shells on many points ---------------- */

// --- fast AO values on arbitrary points (shell-major, vectorized) ----------
Eigen::MatrixXd ao_values_at_points(
    const std::vector<libint2::Shell>& shells,
    const std::vector<Eigen::Vector3d>& points,
    int nthreads) {

  const size_t npts = points.size();
  const size_t nao  = nbasis(shells);
  Eigen::MatrixXd A(nao, npts);
  A.setZero();

  // Flatten points into three Eigen vectors for cache-friendly math
  Eigen::VectorXd Px(npts), Py(npts), Pz(npts);
  for (size_t i = 0; i < npts; ++i) {
    Px[i] = points[i][0];
    Py[i] = points[i][1];
    Pz[i] = points[i][2];
  }

  const auto s2bf = map_shell_to_basis_function(shells);

  #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
  for (ptrdiff_t s = 0; s < (ptrdiff_t)shells.size(); ++s) {
    const auto& sh = shells[(size_t)s];
    const int l = sh.contr[0].l;
    const size_t nfunc = sh.size();
    const double Cx = sh.O[0], Cy = sh.O[1], Cz = sh.O[2];

    // Coordinates relative to shell center
    Eigen::ArrayXd x = Px.array() - Cx;
    Eigen::ArrayXd y = Py.array() - Cy;
    Eigen::ArrayXd z = Pz.array() - Cz;
    Eigen::ArrayXd xx = x*x, yy = y*y, zz = z*z;
    Eigen::ArrayXd r2 = xx + yy + zz;

    // Radial contraction: sum_p coeff_p * exp(-alpha_p * r^2)
    // (Nl = 1.0 here to avoid double normalization)
    Eigen::ArrayXd radial = Eigen::ArrayXd::Zero(npts);
    for (size_t p = 0; p < sh.nprim(); ++p) {
      const double a = sh.alpha[p];
      const double c = sh.contr[0].coeff[p];
      radial += c * (-a * r2).exp();
    }

    // Solid-harmonic polynomials (SO order) multiplied by 'radial'
    Eigen::MatrixXd block(nfunc, npts);
    if (l == 0) {
      block.row(0) = radial.matrix();
    } else if (l == 1) {
      // p: [y, z, x]
      block.row(0) = (radial * y).matrix();
      block.row(1) = (radial * z).matrix();
      block.row(2) = (radial * x).matrix();
    } else if (l == 2) {
      // d (SO): [xy, yz, (3z^2 - r^2), zx, (x^2 - y^2)]
      block.row(0) = (radial * (x*y)).matrix();
      block.row(1) = (radial * (y*z)).matrix();
      block.row(2) = (radial * (3.0*zz - (xx+yy+zz))).matrix();
      block.row(3) = (radial * (z*x)).matrix();
      block.row(4) = (radial * (xx - yy)).matrix();
    } else if (l == 3) {
      // f (SO): [ y(3x^2-y^2), 2xyz, y(5z^2 - r^2), z(5z^2 - 3r^2),
      //           x(5z^2 - r^2), z(x^2 - y^2), x(x^2 - 3y^2) ]
      Eigen::ArrayXd r2 = (xx + yy + zz);
      block.row(0) = (radial * (y * (3.0*xx - yy))).matrix();
      block.row(1) = (radial * (2.0*x*y*z)).matrix();
      block.row(2) = (radial * (y * (5.0*zz - r2))).matrix();
      block.row(3) = (radial * (z * (5.0*zz - 3.0*r2))).matrix();
      block.row(4) = (radial * (x * (5.0*zz - r2))).matrix();
      block.row(5) = (radial * (z * (xx - yy))).matrix();
      block.row(6) = (radial * (x * (xx - 3.0*yy))).matrix();
    } else {
      continue;
    }
    
    const size_t bf0 = s2bf[(size_t)s];
    A.middleRows((Eigen::Index)bf0, (Eigen::Index)nfunc) = block;
  }

  return A;
}

// --- project MOs at arbitrary points (tiled, no A materialization) ---------
// Psi(nm, npts) = sum_shells [ C_rows(l)^H(nm x nfunc) · block(nfunc x tlen) ]
// where 'block' are solid-harmonic AO values for this shell over a tile.
// Policy: hand-evaluated AOs use Nl = 1.0 (avoid double normalization).
Eigen::MatrixXd project_mos_at_points(
    const std::vector<libint2::Shell>& shells,
    const Eigen::MatrixXcd& Csub,                 // (nao, nm)
    const std::vector<Eigen::Vector3d>& points,
    const std::string& part,
    int nthreads) {

  const size_t npts = points.size();
  const size_t nm   = static_cast<size_t>(Csub.cols());

  // Flatten points once
  Eigen::VectorXd Px(npts), Py(npts), Pz(npts);
  for (size_t i = 0; i < npts; ++i) {
    Px[i] = points[i][0];
    Py[i] = points[i][1];
    Pz[i] = points[i][2];
  }

  const auto s2bf = map_shell_to_basis_function(shells);

  Eigen::MatrixXd Psi_all((Eigen::Index)nm, (Eigen::Index)npts);
  Psi_all.setZero();

  // Tile length (tune if you like). 4096–8192 is usually sweet on modern CPUs.
  const ptrdiff_t TL = 4096;

  #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
  for (ptrdiff_t t0 = 0; t0 < (ptrdiff_t)npts; t0 += TL) {
    const ptrdiff_t tlen = std::min<ptrdiff_t>(TL, (ptrdiff_t)npts - t0);

    // Tile coords (arrays)
    Eigen::ArrayXd x = Px.segment(t0, tlen).array();
    Eigen::ArrayXd y = Py.segment(t0, tlen).array();
    Eigen::ArrayXd z = Pz.segment(t0, tlen).array();

    // Local accumulator for this tile (real; we’ll re-do complex if needed)
    Eigen::MatrixXd Psi_tile = Eigen::MatrixXd::Zero((Eigen::Index)nm, tlen);

    // Loop shells
    for (size_t s = 0; s < shells.size(); ++s) {
      const auto& sh = shells[s];
      const int l = sh.contr[0].l;
      const size_t nfunc = sh.size();
      if (l > 3) continue; // up to f, per your ask

      // Shifted coordinates relative to shell center
      const double Cx = sh.O[0], Cy = sh.O[1], Cz = sh.O[2];
      Eigen::ArrayXd xs = x - Cx;
      Eigen::ArrayXd ys = y - Cy;
      Eigen::ArrayXd zs = z - Cz;

      Eigen::ArrayXd xxs = xs*xs, yys = ys*ys, zzs = zs*zs;
      Eigen::ArrayXd r2s = xxs + yys + zzs;
      Eigen::ArrayXd rr  = r2s.sqrt();               // r
      Eigen::ArrayXd rl  = Eigen::ArrayXd::Ones(tlen);
      if (l == 1) rl = rr;
      else if (l == 2) rl = rr.square();
      else if (l == 3) rl = rr*rr.square();

      // Radial contraction with Nl = 1.0 (avoid double normalization)
      Eigen::ArrayXd radial = Eigen::ArrayXd::Zero(tlen);
      for (size_t p = 0; p < sh.nprim(); ++p) {
        const double a = sh.alpha[p];
        const double c = sh.contr[0].coeff[p];
        radial += c * (-a * r2s).exp();
      }
      radial *= rl;   // multiply r^l

      // Angular part in Libint SO order — same polynomials as rsh_array_*,
      // but vectorized (no per-point loops). Matches your definitions above.

      Eigen::MatrixXd block((Eigen::Index)nfunc, tlen);

      if (l == 0) {
        // Y00 ~ const
        block.row(0) = radial.matrix();
      }
      else if (l == 1) {
        // p (SO): [y, z, x]
        block.row(0) = (radial * ys).matrix();
        block.row(1) = (radial * zs).matrix();
        block.row(2) = (radial * xs).matrix();
      }
      else if (l == 2) {
        // d (SO): [xy, yz, (3z^2 - r^2), zx, (x^2 - y^2)]
        block.row(0) = (radial * (xs*ys)).matrix();
        block.row(1) = (radial * (ys*zs)).matrix();
        block.row(2) = (radial * (3.0*zzs - (xxs + yys + zzs))).matrix();
        block.row(3) = (radial * (zs*xs)).matrix();
        block.row(4) = (radial * (xxs - yys)).matrix();
      }
      else { // l == 3
        // f (SO): using the same family you encoded in rsh_array_l3 (unit-sphere forms)
        // Here we need the *solid* harmonics (no 1/r factors), so we multiply
        // by r^3 via 'radial *= rl' above and then use the polynomials
        // expressed in x,y,z directly:
        // SO order: m = -3,-2,-1,0,+1,+2,+3  → 7 functions
        // y(3x^2 - y^2), 2xyz, y(5z^2 - r^2), z(5z^2 - 3r^2),
        // x(5z^2 - r^2), z(x^2 - y^2), x(x^2 - 3y^2)
        Eigen::ArrayXd r2 = (xxs + yys + zzs);
        block.row(0) = (radial * (ys * (3.0*xxs - yys))).matrix();
        block.row(1) = (radial * (2.0*xs*ys*zs)).matrix();
        block.row(2) = (radial * (ys * (5.0*zzs - r2))).matrix();
        block.row(3) = (radial * (zs * (5.0*zzs - 3.0*r2))).matrix();
        block.row(4) = (radial * (xs * (5.0*zzs - r2))).matrix();
        block.row(5) = (radial * (zs * (xxs - yys))).matrix();
        block.row(6) = (radial * (xs * (xxs - 3.0*yys))).matrix();
      }

      // Accumulate: Psi_tile += C_rows^H * block
      const size_t bf0 = s2bf[s];
      const Eigen::Index nfunc_i = (Eigen::Index)nfunc;
      // C_rows: (nfunc, nm)
      Eigen::MatrixXcd C_rows = Csub.middleRows((Eigen::Index)bf0, nfunc_i);
      // (nm, tlen) += (nm, nfunc) * (nfunc, tlen)
      Psi_tile.noalias() += (C_rows.adjoint() * block).real();
    } // shells

    // If the user requested imag/abs/abs2, recompute in complex for this tile.
    if (part != "real") {
      Eigen::MatrixXcd Psi_c = Eigen::MatrixXcd::Zero((Eigen::Index)nm, tlen);

      for (size_t s = 0; s < shells.size(); ++s) {
        const auto& sh = shells[s];
        const int l = sh.contr[0].l;
        const size_t nfunc = sh.size();
        if (l > 3) continue;

        const double Cx = sh.O[0], Cy = sh.O[1], Cz = sh.O[2];
        Eigen::ArrayXd xs = x - Cx;
        Eigen::ArrayXd ys = y - Cy;
        Eigen::ArrayXd zs = z - Cz;
        Eigen::ArrayXd xxs = xs*xs, yys = ys*ys, zzs = zs*zs;
        Eigen::ArrayXd r2s = xxs + yys + zzs;
        Eigen::ArrayXd rr  = r2s.sqrt();
        Eigen::ArrayXd rl  = Eigen::ArrayXd::Ones(tlen);
        if (l == 1) rl = rr;
        else if (l == 2) rl = rr.square();
        else if (l == 3) rl = rr*rr.square();

        Eigen::ArrayXd radial = Eigen::ArrayXd::Zero(tlen);
        for (size_t p = 0; p < sh.nprim(); ++p) {
          const double a = sh.alpha[p];
          const double c = sh.contr[0].coeff[p];
          radial += c * (-a * r2s).exp();
        }
        radial *= rl;

        Eigen::MatrixXd block((Eigen::Index)nfunc, tlen);
        if (l == 0) {
          block.row(0) = radial.matrix();
        } else if (l == 1) {
          block.row(0) = (radial * ys).matrix();
          block.row(1) = (radial * zs).matrix();
          block.row(2) = (radial * xs).matrix();
        } else if (l == 2) {
          block.row(0) = (radial * (xs*ys)).matrix();
          block.row(1) = (radial * (ys*zs)).matrix();
          block.row(2) = (radial * (3.0*zzs - (xxs + yys + zzs))).matrix();
          block.row(3) = (radial * (zs*xs)).matrix();
          block.row(4) = (radial * (xxs - yys)).matrix();
        } else { // l == 3
          Eigen::ArrayXd r2 = (xxs + yys + zzs);
          block.row(0) = (radial * (ys * (3.0*xxs - yys))).matrix();
          block.row(1) = (radial * (2.0*xs*ys*zs)).matrix();
          block.row(2) = (radial * (ys * (5.0*zzs - r2))).matrix();
          block.row(3) = (radial * (zs * (5.0*zzs - 3.0*r2))).matrix();
          block.row(4) = (radial * (xs * (5.0*zzs - r2))).matrix();
          block.row(5) = (radial * (zs * (xxs - yys))).matrix();
          block.row(6) = (radial * (xs * (xxs - 3.0*yys))).matrix();
        }

        const size_t bf0 = s2bf[s];
        Eigen::MatrixXcd C_rows = Csub.middleRows((Eigen::Index)bf0, (Eigen::Index)nfunc);
        Psi_c.noalias() += (C_rows.adjoint() * block);
      }

      if (part == "imag")
        Psi_tile = Psi_c.imag();
      else if (part == "abs")
        Psi_tile = Psi_c.cwiseAbs().matrix().cast<double>();
      else // abs2
        Psi_tile = Psi_c.cwiseAbs2().matrix().cast<double>();
    }

    // Commit tile (unique columns → no races)
    Psi_all.middleCols(t0, tlen) = Psi_tile;
  }

  return Psi_all;
}




} // namespace licpp
