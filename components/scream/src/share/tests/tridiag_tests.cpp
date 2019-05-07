#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"

#include "share/scream_session.hpp"
#include "share/scream_pack.hpp"
#include "share/util/scream_arch.hpp"

template <typename TridiagDiag>
KOKKOS_INLINE_FUNCTION
void fill_tridiag_matrix (TridiagDiag dl, TridiagDiag d, TridiagDiag du,
                          const int& seed) {
  const int nrow = d.extent_int(0);
  const int nprob = d.extent_int(1);

  assert(dl.extent_int(0) == nrow);
  assert(du.extent_int(0) == nrow);

  for (int p = 0; p < nprob; ++p)
    for (int i = 0; i < nrow; ++i) {
      const int k = seed + p + i;
      dl(i,p) = (k % 5 == 0 ? -1 : 1) * 1.3 * (0.1 + ((k*k) % 11));
      du(i,p) = (k % 7 == 0 ? -1 : 1) * 1.7 * (0.2 + ((k*k) % 13));
      d (i,p) = ((k % 3 == 0 ? -1 : 1) *
                 (0.7 + std::abs(dl(i,p)) + std::abs(du(i,p)) + (k % 17)));
    }
}

template <typename DataArray>
KOKKOS_INLINE_FUNCTION
void fill_data_matrix (DataArray X, const int& seed) {
  const int nrow = X.extent_int(0);
  const int nrhs = X.extent_int(1);

  for (int i = 0; i < nrow; ++i)
    for (int j = 0; j < nrhs; ++j)
      X(i,j) = (((7*i + 11*j + 3*i*j) % 3 == 0 ? -1 : 1) *
                1.7 * ((17*(i - 19) + 13*(j - 11) + 5*(i - 5)*(j - 7) + seed) % 47));
}

template <typename TridiagDiag, typename XArray, typename YArray>
KOKKOS_INLINE_FUNCTION
int matvec (TridiagDiag dl, TridiagDiag d, TridiagDiag du, XArray X, YArray Y) {
  const int nrow = d.extent_int(0);
  const int nrhs = X.extent_int(1);
  const int nA = dl.extent_int(2);

  assert(dl.extent_int(0) == nrow);
  assert(du.extent_int(0) == nrow);
  assert(X .extent_int(0) == nrow);
  assert(Y .extent_int(0) == nrow);
  assert(Y .extent_int(1) == nrhs);
  assert(nA == 1 || nA == nrhs);
  assert(d .extent_int(2) == nA);
  assert(du.extent_int(2) == nA);

  const auto dcol = [&] (const int& j) -> int { return nA == nrhs ? j : 0; };

  if (nrow == 1) {
    for (int j = 0; j < nrhs; ++j) {
      const int aj = dcol(j);
      Y(0,j) = d(0,aj) * X(0,j);
    }
    return 0;
  }

  for (int j = 0; j < nrhs; ++j) {
    const int aj = dcol(j);
    Y(0,j) = d(0,aj) * X(0,j) + du(0,aj) * X(1,j);
  }
  for (int i = 1; i < nrow-1; ++i)
    for (int j = 0; j < nrhs; ++j) {
      const int aj = dcol(j);
      Y(i,j) = (dl(i,aj) * X(i-1,j) +
                d (i,aj) * X(i  ,j) +
                du(i,aj) * X(i+1,j));
    }
  const int i = nrow-1;
  for (int j = 0; j < nrhs; ++j) {
    const int aj = dcol(j);
    Y(i,j) = dl(i,aj) * X(i-1,j) + d(i,aj) * X(i,j);
  }

  return 0;  
}

template <typename Array>
scream::Real reldif (const Array& a, const Array& b) {
  assert(a.extent_int(0) == b.extent_int(0));
  assert(a.extent_int(1) == b.extent_int(1));
  assert(a.rank == 2);
  assert(b.rank == 2);
  scream::Real num = 0, den = 0;
  for (int i = 0; i < a.extent_int(0); ++i)
    for (int j = 0; j < a.extent_int(1); ++j) {
      num = std::max(num, std::abs(a(i,j) - b(i,j)));
      den = std::max(den, std::abs(a(i,j)));
    }
  return num/den;
}

struct Solver {
  enum Enum { thomas_scalar, thomas_pack, cr_scalar, cr_pack, error };
  static std::string convert (Enum e) {
    switch (e) {
    case thomas_scalar: return "thomas_scalar";
    case cr_scalar: return "cr_scalar";
    case thomas_pack: return "thomas_pack";
    case cr_pack: return "cr_pack";
    default:
      assert(0);
      return "";
    }
  }
  static Enum convert (const std::string& s) {
    if (s == "thomas_scalar") return thomas_scalar;
    if (s == "cr_scalar") return cr_scalar;
    if (s == "thomas_pack") return thomas_pack;
    if (s == "cr_pack") return cr_pack;
    return error;
  }

  static Enum all[];
};

Solver::Enum Solver::all[] = { thomas_scalar, thomas_pack, cr_scalar, cr_pack };

struct TestConfig {
  using BulkLayout = Kokkos::LayoutRight;
  using TeamLayout = Kokkos::LayoutRight;

  Solver::Enum solver;
  int n_kokkos_thread, n_kokkos_vec;
};

template <typename ScalarType>
using TridiagArray = Kokkos::View<ScalarType***, TestConfig::TeamLayout>;
template <typename ScalarType>
using DataArray = Kokkos::View<ScalarType**, TestConfig::TeamLayout>;

void run_test (const TestConfig& tc) {
  using Kokkos::create_mirror_view;
  using Kokkos::deep_copy;
  using Kokkos::subview;
  using Kokkos::ALL;
  using scream::Real;

  using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  using MT = typename TeamPolicy::member_type;
  TeamPolicy policy(1, tc.n_kokkos_thread, tc.n_kokkos_vec);

  for (const int nrow : {1,2,3,4, 8,10,16, 32,43, 63,64,65, 111,128,129, 8192}) {
    const int nrhs_max = 60;
    const int nrhs_inc = 11;
    for (int nrhs = 1; nrhs <= nrhs_max; nrhs += nrhs_inc) {
      for (const bool A_many : {false, true}) {
        const int nprob = A_many ? nrhs : 1;

        TridiagArray<Real>
          A("A", nrow, 3, nprob),
          Acopy("A", A.extent(0), A.extent(1), A.extent(2));
        DataArray<Real>
          B("B", A.extent(0), nrhs), X("X", B.extent(0), B.extent(1)),
          Y("Y", X.extent(0), X.extent(1));
        auto Am = create_mirror_view(A);
        auto Bm = create_mirror_view(B);
        {
          const auto dl = subview(Am, ALL(), 0, ALL());
          const auto d  = subview(Am, ALL(), 1, ALL());
          const auto du = subview(Am, ALL(), 2, ALL());
          fill_tridiag_matrix(dl, d, du, nrhs);
        }
        fill_data_matrix(Bm, nrhs);
        deep_copy(A, Am);
        deep_copy(B, Bm);
        deep_copy(Acopy, A);
        deep_copy(X, B);

        switch (tc.solver) {
        case Solver::thomas_scalar: {
        } break;
        case Solver::thomas_pack: {
        } break;
        case Solver::cr_scalar: {
        } break;
        case Solver::cr_pack: {
        } break;
        default: scream_require_msg(false, "Not a solver: " << tc.solver);
        }

        Real re; {
          auto Acopym = create_mirror_view(Acopy);
          const auto dl = subview(Acopym, ALL(), 0, ALL());
          const auto d  = subview(Acopym, ALL(), 1, ALL());
          const auto du = subview(Acopym, ALL(), 2, ALL());
          auto Xm = create_mirror_view(X);
          auto Ym = create_mirror_view(Y);
          deep_copy(Acopym, Acopy);
          deep_copy(Xm, X);
          matvec(dl, d, du, Xm, Ym);
          re = reldif(Bm, Ym);
        }
        REQUIRE(re <= 50*std::numeric_limits<Real>::epsilon());
      }
    }
  }
}

TEST_CASE("tridiag", "correctness") {
  for (const auto solver : Solver::all) {
    TestConfig tc;
    tc.solver = solver;
    if (scream::util::OnGpu<Kokkos::DefaultExecutionSpace>::value) {
      tc.n_kokkos_vec = 1;
      for (const int n_kokkos_thread : {128, 256, 512}) {
        tc.n_kokkos_thread = n_kokkos_thread;
        run_test(tc);
      }
      tc.n_kokkos_vec = 32;
      for (const int n_kokkos_thread : {4}) {
        tc.n_kokkos_thread = n_kokkos_thread;
        run_test(tc);
      }
    } else {
      const int concurrency = Kokkos::DefaultExecutionSpace::concurrency();
      const int n_kokkos_thread = concurrency;
      for (const int n_kokkos_vec : {1, 2}) {
        tc.n_kokkos_thread = n_kokkos_thread;
        tc.n_kokkos_vec = n_kokkos_vec;
        run_test(tc);
      }
    }
  }
}

int main (int argc, char **argv) {
  int num_failed = 0;
  scream::initialize_scream_session(argc, argv); {
    if (argc > 1) {
      // Performance test.
      // TODO
    } else {
      // Correctness tests.
      num_failed = Catch::Session().run(argc, argv);
    }
  } scream::finalize_scream_session();
  return num_failed != 0 ? 1 : 0;
}
