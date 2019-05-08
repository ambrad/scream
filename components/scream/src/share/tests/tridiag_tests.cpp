#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"

#include "share/util/scream_tridiag.hpp"

#include "share/scream_session.hpp"
#include "share/scream_pack.hpp"
#include "share/scream_pack_kokkos.hpp"
#include "share/util/scream_arch.hpp"
#include "share/util/scream_utils.hpp"

#include <chrono>

template <typename TridiagDiag>
KOKKOS_INLINE_FUNCTION
void fill_tridiag_matrix (TridiagDiag dl, TridiagDiag d, TridiagDiag du,
                          const int& nprob, const int& seed) {
  const int nrow = d.extent_int(0);

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
int matvec (TridiagDiag dl, TridiagDiag d, TridiagDiag du, XArray X, YArray Y,
            const int nprob, const int nrhs) {
  const int nrow = d.extent_int(0);

  assert(dl.extent_int(0) == nrow);
  assert(du.extent_int(0) == nrow);
  assert(X .extent_int(0) == nrow);
  assert(X .extent_int(1) >= nrhs);
  assert(Y .extent_int(0) == nrow);
  assert(Y .extent_int(1) == X.extent_int(1));
  assert(dl.extent_int(1) >= nprob);
  assert(nprob == 1 || nprob == nrhs);
  assert(d .extent_int(1) == dl.extent_int(1));
  assert(du.extent_int(1) == dl.extent_int(1));

  const auto dcol = [&] (const int& j) -> int { return nprob > 1 ? j : 0; };

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
scream::Real reldif (const Array& a, const Array& b, const int nrhs) {
  assert(a.extent_int(0) == b.extent_int(0));
  assert(a.extent_int(1) == b.extent_int(1));
  assert(a.rank == 2);
  assert(b.rank == 2);
  scream::Real num = 0, den = 0;
  for (int i = 0; i < a.extent_int(0); ++i)
    for (int j = 0; j < nrhs; ++j) {
      if (std::isnan(a(i,j)) || std::isnan(b(i,j)) ||
          std::isinf(a(i,j)) || std::isinf(b(i,j))) {
        return std::numeric_limits<scream::Real>::infinity();
      }
      num = std::max(num, std::abs(a(i,j) - b(i,j)));
      den = std::max(den, std::abs(a(i,j)));
    }
  return num/den;
}

struct Solver {
  enum Enum { thomas_team_scalar, thomas_team_pack,
              thomas_scalar, thomas_pack,
              cr_scalar,
              error };

  static std::string convert (Enum e) {
    switch (e) {
    case thomas_team_scalar: return "thomas_team_scalar";
    case thomas_team_pack: return "thomas_team_pack";
    case thomas_scalar: return "thomas_scalar";
    case thomas_pack: return "thomas_pack";
    case cr_scalar: return "cr_scalar";
    default: scream_require_msg(false, "Not a valid solver: " << e);
    }
  }

  static Enum convert (const std::string& s) {
    if (s == "thomas_team_scalar") return thomas_team_scalar;
    if (s == "thomas_team_pack") return thomas_team_pack;
    if (s == "thomas_scalar") return thomas_scalar;
    if (s == "thomas_pack") return thomas_pack;
    if (s == "cr_scalar") return cr_scalar;
    return error;
  }

  static Enum all[];
};

Solver::Enum Solver::all[] = { thomas_team_scalar, thomas_team_pack,
                               thomas_scalar, thomas_pack,
                               cr_scalar };

namespace test_correct {
struct TestConfig {
  using TeamLayout = Kokkos::LayoutRight;

  Solver::Enum solver;
  int n_kokkos_thread, n_kokkos_vec;
};

template <typename TridiagArray>
KOKKOS_INLINE_FUNCTION
Kokkos::View<typename TridiagArray::value_type*>
get_diag (const TridiagArray& A, const int diag_idx) {
  assert(A.extent_int(2) == 1);
  return Kokkos::View<typename TridiagArray::value_type*>(
    &A.impl_map().reference(diag_idx, 0, 0),
    A.extent_int(1));
}

template <typename TridiagArray>
Kokkos::View<typename TridiagArray::value_type**, TestConfig::TeamLayout>
KOKKOS_INLINE_FUNCTION
get_diags (const TridiagArray& A, const int diag_idx) {
  return Kokkos::View<typename TridiagArray::value_type**, TestConfig::TeamLayout>(
    &A.impl_map().reference(diag_idx, 0, 0),
    A.extent_int(1), A.extent_int(2));
}

template <typename DataArray>
KOKKOS_INLINE_FUNCTION
Kokkos::View<typename DataArray::value_type*>
get_x (const DataArray& X) {
  assert(X.extent_int(1) == 1);
  return Kokkos::View<typename DataArray::value_type*>(
    &X.impl_map().reference(0, 0), X.extent_int(0));
}

template <typename Scalar>
using TridiagArray = Kokkos::View<Scalar***, TestConfig::TeamLayout>;
template <typename Scalar>
using DataArray = Kokkos::View<Scalar**, TestConfig::TeamLayout>;

template <bool same_pack_size, typename APack, typename DataPack>
struct Solve;

template <typename APack, typename DataPack>
struct Solve<true, APack, DataPack> {
  static void run (const TestConfig& tc, TridiagArray<APack>& A, DataArray<DataPack>& X,
                   const int nprob, const int nrhs) {
    using Kokkos::subview;
    using Kokkos::ALL;
    using scream::pack::scalarize;

    assert(nrhs > 1 || DataPack::n == 1);
    assert(nrhs > 1 || X.extent_int(2) == 1);
    assert(nprob > 1 || APack::n == 1);
    assert(nprob > 1 || A.extent_int(2) == 1);

    using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using MT = typename TeamPolicy::member_type;
    TeamPolicy policy(1, tc.n_kokkos_thread, tc.n_kokkos_vec);

    switch (tc.solver) {
    case Solver::thomas_team_scalar:
    case Solver::thomas_team_pack: {
      const auto As = scalarize(A);
      const auto Xs = scalarize(X);
      const auto f = KOKKOS_LAMBDA (const MT& team) {
        const auto dl = get_diag(As, 0);
        const auto d  = get_diag(As, 1);
        const auto du = get_diag(As, 2);
        if (tc.solver == Solver::thomas_team_scalar)
          scream::tridiag::thomas(team, dl, d, du, Xs);
        else
          scream::tridiag::thomas(team, dl, d, du, X);
      };
      Kokkos::parallel_for(policy, f);
    } break;
    case Solver::thomas_scalar: {
      if (nprob == 1) {
        if (nrhs == 1) {
          const auto As = scalarize(A);
          const auto Xs = scalarize(X);
          const auto f = KOKKOS_LAMBDA (const MT& team) {
            const auto single = [&] () {
              const auto dl = get_diag(As, 0);
              const auto d  = get_diag(As, 1);
              const auto du = get_diag(As, 2);
              const auto x  = get_x(Xs);
              scream::tridiag::thomas(dl, d, du, x);
            };
            Kokkos::single(Kokkos::PerTeam(team), single);
          };
          Kokkos::parallel_for(policy, f);
        } else {
          const auto As = scalarize(A);
          const auto Xs = scalarize(X);
          const auto f = KOKKOS_LAMBDA (const MT& team) {
            const auto single = [&] () {
              const auto dl = get_diag(As, 0);
              const auto d  = get_diag(As, 1);
              const auto du = get_diag(As, 2);
              scream::tridiag::thomas(dl, d, du, Xs);
            };
            Kokkos::single(Kokkos::PerTeam(team), single);
          };
          Kokkos::parallel_for(policy, f);
        }
      } else {
        const auto As = scalarize(A);
        const auto Xs = scalarize(X);
        const auto f = KOKKOS_LAMBDA (const MT& team) {
          const auto single = [&] () {
            const auto dl = get_diags(As, 0);
            const auto d  = get_diags(As, 1);
            const auto du = get_diags(As, 2);
            scream::tridiag::thomas(dl, d, du, Xs);
          };
          Kokkos::single(Kokkos::PerTeam(team), single);
        };
        Kokkos::parallel_for(policy, f);
      }
    } break;
    case Solver::thomas_pack: {
      if (nprob == 1) {
        const auto As = scalarize(A);
        const auto f = KOKKOS_LAMBDA (const MT& team) {
          const auto single = [&] () {
            const auto dl = get_diag(As, 0);
            const auto d  = get_diag(As, 1);
            const auto du = get_diag(As, 2);
            scream::tridiag::thomas(dl, d, du, X);
          };
          Kokkos::single(Kokkos::PerTeam(team), single);
        };
        Kokkos::parallel_for(policy, f);
      } else {
        const auto f = KOKKOS_LAMBDA (const MT& team) {
          const auto single = [&] () {
            const auto dl = get_diags(A, 0);
            const auto d  = get_diags(A, 1);
            const auto du = get_diags(A, 2);
            scream::tridiag::thomas(dl, d, du, X);
          };
          Kokkos::single(Kokkos::PerTeam(team), single);
        };
        Kokkos::parallel_for(policy, f);
      }
    } break;
    case Solver::cr_scalar: {
      if (nprob == 1) {
        if (nrhs == 1) {
          const auto As = scalarize(A);
          const auto Xs = scalarize(X);
          const auto f = KOKKOS_LAMBDA (const MT& team) {
            const auto dl = get_diag(As, 0);
            const auto d  = get_diag(As, 1);
            const auto du = get_diag(As, 2);
            const auto x  = get_x(Xs);
            scream::tridiag::cr(team, dl, d, du, x);
          };
          Kokkos::parallel_for(policy, f);
        } else {
          const auto As = scalarize(A);
          const auto Xs = scalarize(X);
          const auto f = KOKKOS_LAMBDA (const MT& team) {
            const auto dl = get_diag(As, 0);
            const auto d  = get_diag(As, 1);
            const auto du = get_diag(As, 2);
            scream::tridiag::cr(team, dl, d, du, Xs);
          };
          Kokkos::parallel_for(policy, f);
        }
      } else {
        const auto As = scalarize(A);
        const auto Xs = scalarize(X);
        const auto f = KOKKOS_LAMBDA (const MT& team) {
          const auto dl = get_diags(As, 0);
          const auto d  = get_diags(As, 1);
          const auto du = get_diags(As, 2);
          scream::tridiag::cr(team, dl, d, du, Xs);
        };
        Kokkos::parallel_for(policy, f);
      }
    } break;
    default:
      scream_require_msg(false, "Same pack size: " << Solver::convert(tc.solver));
    }
  }
};

template <typename APack, typename DataPack>
struct Solve<false, APack, DataPack> {
  static void run (const TestConfig& tc, TridiagArray<APack>& A, DataArray<DataPack>& X,
                   const int nprob, const int nrhs) {
    using Kokkos::subview;
    using Kokkos::ALL;
    using scream::pack::scalarize;

    using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using MT = typename TeamPolicy::member_type;
    TeamPolicy policy(1, tc.n_kokkos_thread, tc.n_kokkos_vec);

    assert(nrhs > 1 || DataPack::n == 1);
    assert(nrhs > 1 || X.extent_int(2) == 1);
    assert(nprob == 1);

    switch (tc.solver) {
    case Solver::thomas_team_pack: {
      const auto As = scalarize(A);
      const auto Xs = scalarize(X);
      const auto f = KOKKOS_LAMBDA (const MT& team) {
        const auto dl = get_diag(As, 0);
        const auto d  = get_diag(As, 1);
        const auto du = get_diag(As, 2);
        if (tc.solver == Solver::thomas_team_scalar)
          scream::tridiag::thomas(team, dl, d, du, Xs);
        else
          scream::tridiag::thomas(team, dl, d, du, X);
      };
      Kokkos::parallel_for(policy, f);
    } break;
    case Solver::thomas_pack: {
      const auto As = scalarize(A);
      const auto f = KOKKOS_LAMBDA (const MT& team) {
        const auto single = [&] () {
          const auto dl = get_diag(As, 0);
          const auto d  = get_diag(As, 1);
          const auto du = get_diag(As, 2);
          scream::tridiag::thomas(dl, d, du, X);
        };
        Kokkos::single(Kokkos::PerTeam(team), single);
      };
      Kokkos::parallel_for(policy, f);
    } break;
    default:
      scream_require_msg(false, "Different pack size: " << Solver::convert(tc.solver));
    }
  }
};

template <int A_pack_size, int data_pack_size>
void run_test (const TestConfig& tc) {
  using Kokkos::create_mirror_view;
  using Kokkos::deep_copy;
  using Kokkos::subview;
  using Kokkos::ALL;
  using scream::Real;
  using scream::pack::npack;
  using scream::pack::scalarize;
  using APack = scream::pack::Pack<Real, A_pack_size>;
  using DataPack = scream::pack::Pack<Real, data_pack_size>;

#if 1
  const int nrows[] = {1,2,3,4,5, 8,10,16, 32,43, 63,64,65, 111,128,129, 2048};
  const int nrhs_max = 60;
  const int nrhs_inc = 11;
#else
  const int nrows[] = {10};
  const int nrhs_max = 6;
  const int nrhs_inc = 5;  
#endif

  for (const int nrow : nrows) {
    for (int nrhs = 1; nrhs <= nrhs_max; nrhs += nrhs_inc) {
      for (const bool A_many : {false, true}) {
        if (nrhs == 1 && A_many) continue;
        const int nprob = A_many ? nrhs : 1;

        // Skip unsupported solver-problem format combinations.
        if ((tc.solver == Solver::thomas_team_scalar ||
             tc.solver == Solver::thomas_team_pack)
            && nprob > 1)
          continue;

        // Skip combinations generated at this and higher levels that Solve::run
        // doesn't support to reduce redundancies.
        if ((nrhs  == 1 && data_pack_size > 1) ||
            (nprob == 1 && A_pack_size    > 1))
          continue;
        if ((tc.solver == Solver::thomas_team_scalar ||
             tc.solver == Solver::thomas_scalar ||
             tc.solver == Solver::cr_scalar) &&
            data_pack_size > 1)
          continue;
        if (static_cast<int>(APack::n) != static_cast<int>(DataPack::n) && nprob > 1)
          continue;

        const int prob_npack = npack<APack>(nprob);
        const int rhs_npack = npack<DataPack>(nrhs);

        TridiagArray<APack>
          A("A", 3, nrow, prob_npack),
          Acopy("A", A.extent(0), A.extent(1), A.extent(2));
        DataArray<DataPack>
          B("B", nrow, rhs_npack), X("X", B.extent(0), B.extent(1)),
          Y("Y", X.extent(0), X.extent(1));
        const auto Am = create_mirror_view(A);
        const auto Bm = create_mirror_view(B);
        {
          const auto As = scalarize(Am);
          const auto dl = subview(As, 0, ALL(), ALL());
          const auto d  = subview(As, 1, ALL(), ALL());
          const auto du = subview(As, 2, ALL(), ALL());
          fill_tridiag_matrix(dl, d, du, nprob, nrhs /* seed */);
        }
        fill_data_matrix(scalarize(Bm), nrhs);
        deep_copy(A, Am);
        deep_copy(B, Bm);
        deep_copy(Acopy, A);
        deep_copy(X, B);

        Solve<A_pack_size == data_pack_size, APack, DataPack>
          ::run(tc, A, X, nprob, nrhs);

        Real re; {
          const auto Acopym = create_mirror_view(Acopy);
          const auto As = scalarize(Acopym);
          const auto dl = subview(As, 0, ALL(), ALL());
          const auto d  = subview(As, 1, ALL(), ALL());
          const auto du = subview(As, 2, ALL(), ALL());
          const auto Xm = create_mirror_view(X);
          const auto Ym = create_mirror_view(Y);
          deep_copy(Acopym, Acopy);
          deep_copy(Xm, X);
          matvec(dl, d, du, scalarize(Xm), scalarize(Ym), nprob, nrhs);
          re = reldif(scalarize(Bm), scalarize(Ym), nrhs);
        }
        const bool pass = re <= 50*std::numeric_limits<Real>::epsilon();
        std::stringstream ss;
        ss << Solver::convert(tc.solver) << " " << tc.n_kokkos_thread
           << " " << tc.n_kokkos_vec << " | " << nrow << " " << nrhs << " "
           << A_many << " | log10 reldif " << std::log10(re);
        if ( ! pass) std::cout << "FAIL: " << ss.str() << "\n";
        REQUIRE(pass);
        //std::cout << "PASS: " << ss.str() << "\n";
      }
    }
  }
}

template <int A_pack_size, int data_pack_size>
void run_test () {
  for (const auto solver : Solver::all) {
    TestConfig tc;
    tc.solver = solver;
    if (scream::util::OnGpu<Kokkos::DefaultExecutionSpace>::value) {
      tc.n_kokkos_vec = 1;
      for (const int n_kokkos_thread : {128, 256, 512}) {
        tc.n_kokkos_thread = n_kokkos_thread;
        run_test<A_pack_size, data_pack_size>(tc);
      }
      tc.n_kokkos_vec = 32;
      for (const int n_kokkos_thread : {4}) {
        tc.n_kokkos_thread = n_kokkos_thread;
        run_test<A_pack_size, data_pack_size>(tc);
      }
      tc.n_kokkos_vec = 8;
      for (const int n_kokkos_thread : {16}) {
        tc.n_kokkos_thread = n_kokkos_thread;
        run_test<A_pack_size, data_pack_size>(tc);
      }
    } else {
      const int concurrency = Kokkos::DefaultExecutionSpace::concurrency();
      const int n_kokkos_thread = concurrency;
      for (const int n_kokkos_vec : {1, 2}) {
        tc.n_kokkos_thread = n_kokkos_thread;
        tc.n_kokkos_vec = n_kokkos_vec;
        run_test<A_pack_size, data_pack_size>(tc);
      }
    }
  }
}
} // namespace test_correct

TEST_CASE("tridiag", "correctness") {
  test_correct::run_test<1,1>();
  if (SCREAM_PACK_SIZE > 1) {
    test_correct::run_test<1, SCREAM_PACK_SIZE>();
    test_correct::run_test<SCREAM_PACK_SIZE, SCREAM_PACK_SIZE>();
  }
}

namespace perf {
void expect_another_arg (int i, int argc) {
  if (i == argc-1)
    throw std::runtime_error("Expected another cmd-line arg.");
}

struct Input {
  struct Solver {
    enum Enum { thomas, cr, error };

    static std::string convert (Enum e) {
      switch (e) {
      case thomas: return "thomas";
      case cr: return "cr";
      default: scream_require_msg(false, "Not a valid solver: " << e);
      }
    }

    static Enum convert (const std::string& s) {
      if (s == "thomas") return thomas;
      if (s == "cr") return cr;
      return error;
    }
  };

  Solver::Enum method;
  int nprob, nrow, nrhs, nwarp;
  bool pack, oneA;

  Input ()
    : method(Solver::cr), nprob(2048), nrow(128), nrhs(43), nwarp(-1),
      pack( ! scream::util::OnGpu<Kokkos::DefaultExecutionSpace>::value),
      oneA(false)
  {}

  bool parse (int argc, char** argv) {
    using scream::util::eq;
    for (int i = 1; i < argc; ++i) {
      if (eq(argv[i], "-m", "--method")) {
        expect_another_arg(i, argc);
        method = Input::Solver::convert(argv[++i]);
        if (method == Input::Solver::error) {
          std::cout << "Not a solver: " << argv[i] << "\n";
          return false;
        }
      } else if (eq(argv[i], "-np", "--nprob")) {
        expect_another_arg(i, argc);
        nprob = std::atoi(argv[++i]);
      } else if (eq(argv[i], "-nr", "--nrow")) {
        expect_another_arg(i, argc);
        nrow = std::atoi(argv[++i]);
      } else if (eq(argv[i], "-nc", "--nrhs")) {
        expect_another_arg(i, argc);
        nrhs = std::atoi(argv[++i]);
      } else if (eq(argv[i], "-1a", "--oneA")) {
        oneA = true;
      } else if (eq(argv[i], "-nw", "--nwarp")) {
        expect_another_arg(i, argc);
        nwarp = std::atoi(argv[++i]);
      } else if (eq(argv[i], "-nop", "--nopack")) {
        pack = false;
      } else {
        std::cout << "Unexpected arg: " << argv[i] << "\n";
        return false;
      }
    }
    return true;
  }
};

std::string string (const Input& in, const int& nwarp) {
  std::stringstream ss;
  ss << "run: solver " << Input::Solver::convert(in.method)
     << " pack " << in.pack
     << " nprob " << in.nprob
     << " nrow " << in.nrow
     << " nA " << (in.oneA ? 1 : in.nrhs)
     << " nrhs " << in.nrhs
     << " nwarp " << nwarp << "\n";
  return ss.str();
}

template <typename ST, typename DT>
void pack_data (const ST& a, DT b) {
  using Pack = typename DT::non_const_value_type;
  static_assert(Pack::packtag, "DT value type must be Pack");
  static_assert(std::is_same<typename ST::non_const_value_type,
                             typename Pack::scalar>::value,
                "ST value and DT::Pack::scalar types must be the same");
  const int np = a.extent_int(0);
  const int m = a.extent_int(1), n = a.extent_int(2), mn = m*n;
  assert(b.extent_int(0) == np);
  assert(b.extent_int(1) == m);
  assert(b.extent_int(2) == (n + Pack::n - 1)/Pack::n);
  auto f = KOKKOS_LAMBDA (const int i) {
    const int prob = i / mn;
    const int row = (i % mn) / n;
    const int col = i % n;
    b(prob, row, col / Pack::n)[col % Pack::n] = a(prob,row,col);
  };
  Kokkos::parallel_for(np*m*n, f);
}

template <typename ST, typename DT>
void unpack_data (const ST& a, DT b) {
  using Pack = typename ST::non_const_value_type;
  static_assert(Pack::packtag, "ST value type must be Pack");
  static_assert(std::is_same<typename DT::non_const_value_type,
                             typename Pack::scalar>::value,
                "DT value and ST::Pack::scalar types must be the same");
  const int np = b.extent_int(0);
  const int m = b.extent_int(1), n = b.extent_int(2), mn = m*n;
  assert(a.extent_int(0) == np);
  assert(a.extent_int(1) == m);
  assert(a.extent_int(2) == (n + Pack::n - 1)/Pack::n);
  auto f = KOKKOS_LAMBDA (const int i) {
    const int prob = i / mn;
    const int row = (i % mn) / n;
    const int col = i % n;
    b(prob,row,col) = a(prob, row, col / Pack::n)[col % Pack::n];
  };
  Kokkos::parallel_for(np*m*n, f);
}

template <typename ST, typename DT>
void pack_matrix (const ST& a, DT b, const int nrhs) {
  using Pack = typename DT::non_const_value_type;
  static_assert(Pack::packtag, "DT value type must be Pack");
  static_assert(std::is_same<typename ST::non_const_value_type,
                             typename Pack::scalar>::value,
                "ST value and DT::Pack::scalar types must be the same");
  const int np = a.extent_int(0);
  const int nrow = a.extent_int(2);
  assert(b.extent_int(0) == np);
  assert(b.extent_int(1) == 3);
  assert(b.extent_int(2) == nrow);
  assert(b.extent_int(3) == (nrhs + Pack::n - 1)/Pack::n);
  auto f = KOKKOS_LAMBDA (const int i) {
    const int prob = i / (3*nrow*nrhs);
    const int c = (i % (3*nrow*nrhs)) / (nrow*nrhs);
    const int row = (i % (nrow*nrhs)) / nrhs;
    const int rhs = i % nrhs;
    b(prob, c, row, rhs / Pack::n)[rhs % Pack::n] = a(prob,c,row);
  };
  Kokkos::parallel_for(np*3*nrow*nrhs, f);
}

template <typename ST, typename DT>
void pack_scalar_matrix (const ST& a, DT b, const int nrhs) {
  const int np = a.extent_int(0);
  const int nrow = a.extent_int(2);
  assert(b.extent_int(0) == np);
  assert(b.extent_int(1) == 3);
  assert(b.extent_int(2) == nrow);
  assert(b.extent_int(3) == nrhs);
  auto f = KOKKOS_LAMBDA (const int i) {
    const int prob = i / (3*nrow*nrhs);
    const int c = (i % (3*nrow*nrhs)) / (nrow*nrhs);
    const int row = (i % (nrow*nrhs)) / nrhs;
    const int rhs = i % nrhs;
    b(prob,c,row,rhs) = a(prob,c,row);
  };
  Kokkos::parallel_for(np*3*nrow*nrhs, f);
}

using BulkLayout = Kokkos::LayoutRight;
using TeamLayout = Kokkos::LayoutRight;

template <typename TridiagArray>
KOKKOS_INLINE_FUNCTION
Kokkos::View<typename TridiagArray::value_type*>
get_diag (const TridiagArray& A, const int& ip, const int& diag_idx) {
  assert(A.extent_int(3) == 1);
  return Kokkos::View<typename TridiagArray::value_type*>(
    &A.impl_map().reference(ip, diag_idx, 0, 0),
    A.extent_int(2));
}

template <typename TridiagArray>
KOKKOS_INLINE_FUNCTION
Kokkos::View<typename TridiagArray::value_type**, TeamLayout>
get_diags (const TridiagArray& A, const int& ip, const int& diag_idx) {
  return Kokkos::View<typename TridiagArray::value_type**, TeamLayout>(
    &A.impl_map().reference(ip, diag_idx, 0, 0),
    A.extent_int(2), A.extent_int(3));
}

template <typename DataArray>
KOKKOS_INLINE_FUNCTION
Kokkos::View<typename DataArray::value_type*>
get_x (const DataArray& X, const int& ip) {
  assert(X.extent_int(2) == 1);
  return Kokkos::View<typename DataArray::value_type*>(
    &X.impl_map().reference(ip, 0, 0), X.extent_int(1));
}

template <typename DataArray>
KOKKOS_INLINE_FUNCTION
Kokkos::View<typename DataArray::value_type**, TeamLayout>
get_xs (const DataArray& X, const int& ip) {
  return Kokkos::View<typename DataArray::value_type**, TeamLayout>(
    &X.impl_map().reference(ip, 0, 0), X.extent_int(1), X.extent_int(2));
}

template <typename Scalar>
using TridiagArrays = Kokkos::View<Scalar****, BulkLayout>;
template <typename Scalar>
using DataArrays = Kokkos::View<Scalar***, BulkLayout>;

template <typename Real>
void run (const Input& in) {
  using Kokkos::create_mirror_view;
  using Kokkos::deep_copy;
  using Kokkos::subview;
  using Kokkos::ALL;
  using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  using MT = typename TeamPolicy::member_type;
  using Solver = Input::Solver;

  const auto gettime = [&] () {
    return std::chrono::steady_clock::now();
  };
  using TimePoint = decltype(gettime());
  const auto duration = [&] (const TimePoint& t0, const TimePoint& tf) -> double {
    return 1e-6*std::chrono::duration_cast<std::chrono::microseconds>(tf - t0).count();
  };

  const bool on_gpu = scream::util::OnGpu<Kokkos::DefaultExecutionSpace>::value;
  const int nA = in.oneA ? 1 : in.nrhs;

  TridiagArrays<Real> A("A", in.nprob, 3, in.nrow, nA),
    Acopy("Acopy", A.extent(0), A.extent(1), A.extent(2), A.extent(3));
  DataArrays<Real> B("B", in.nprob, in.nrow, in.nrhs),
    X("X", in.nprob, in.nrow, in.nrhs), Y("Y", in.nprob, in.nrow, in.nrhs);
  auto Am = create_mirror_view(A);
  auto Bm = create_mirror_view(B);
  for (int i = 0; i < in.nprob; ++i) {
    const auto dl = subview(Am, i, 0, ALL(), ALL());
    const auto d  = subview(Am, i, 1, ALL(), ALL());
    const auto du = subview(Am, i, 2, ALL(), ALL());
    fill_tridiag_matrix(dl, d, du, nA, i);
    fill_data_matrix(subview(Bm, i, ALL(), ALL()), in.nrhs);
  }
  deep_copy(A, Am);
  deep_copy(B, Bm);
  deep_copy(Acopy, A);
  deep_copy(X, B);

  TeamPolicy policy(in.nprob, on_gpu ? 128 : 1, 1);
  assert(in.nwarp < 0 || ! on_gpu || policy.team_size() == 32*in.nwarp);
  std::cout << string(in, policy.team_size()/32);

  Kokkos::fence();
  TimePoint t0, t1;
  switch (in.method) {
  case Solver::thomas: {
    if (on_gpu) {
      scream_require_msg(
        in.oneA, "On GPU, only 1 A/team is supported in the Thomas algorithm.");
      t0 = gettime();
      const auto f = KOKKOS_LAMBDA (const MT& team) {
        const int ip = team.league_rank();
        const auto dl = get_diag(A, ip, 0);
        const auto d  = get_diag(A, ip, 1);
        const auto du = get_diag(A, ip, 2);
        const auto x = get_xs(X, ip);
        scream::tridiag::thomas(team, dl, d, du, x);
      };
      Kokkos::parallel_for(policy, f);
      Kokkos::fence();
      t1 = gettime();
    } else {
      if (in.pack) {
      } else {
        for (int trial = 0; trial < 2; ++trial) {
          deep_copy(A, Acopy);
          // Kokkos::deep_copy was messing up the timing for some reason. Do it
          // manually.
          const auto dc = KOKKOS_LAMBDA (const MT& team) {
            const auto s = [&] () {
              const int i = team.league_rank();
              for (int r = 0; r < in.nrow; ++r)
                for (int c = 0; c < in.nrhs; ++c)
                  X(i,r,c) = B(i,r,c);
            };
            Kokkos::single(Kokkos::PerTeam(team), s);
          };
          Kokkos::parallel_for(policy, dc);
          Kokkos::fence();
          t0 = gettime();
          if (in.nrhs == 1) {
            assert(in.oneA);
            const auto f = KOKKOS_LAMBDA (const MT& team) {
              const auto single = [&] () {
                const int ip = team.league_rank();
                const auto dl = get_diag(A, ip, 0);
                const auto d  = get_diag(A, ip, 1);
                const auto du = get_diag(A, ip, 2);
                const auto x  = get_x(X, ip);
                scream::tridiag::thomas(dl, d, du, x);
              };
              Kokkos::single(Kokkos::PerTeam(team), single);
            };
            Kokkos::parallel_for(policy, f);
          } else {
            if (in.oneA) {
              const auto f = KOKKOS_LAMBDA (const MT& team) {
                const auto single = [&] () {
                  const int ip = team.league_rank();
                  const auto dl = get_diag(A, ip, 0);
                  const auto d  = get_diag(A, ip, 1);
                  const auto du = get_diag(A, ip, 2);
                  const auto x  = get_xs(X, ip);
                  scream::tridiag::thomas(dl, d, du, x);
                };
                Kokkos::single(Kokkos::PerTeam(team), single);
              };
              Kokkos::parallel_for(policy, f);
            } else {
              const auto f = KOKKOS_LAMBDA (const MT& team) {
                const auto single = [&] () {
                  const int ip = team.league_rank();
                  const auto dl = get_diags(A, ip, 0);
                  const auto d  = get_diags(A, ip, 1);
                  const auto du = get_diags(A, ip, 2);
                  const auto x  = get_xs(X, ip);
                  assert(x.extent_int(1) == in.nrhs);
                  assert(d.extent_int(1) == in.nrhs);
                  scream::tridiag::thomas(dl, d, du, x);
                };
                Kokkos::single(Kokkos::PerTeam(team), single);
              };
              Kokkos::parallel_for(policy, f);
            }
          }
          Kokkos::fence();
          t1 = gettime();
        }
      }
    }
  } break;
  case Solver::cr: {
    t0 = gettime();
    if (in.nrhs == 1) {
      assert(in.oneA);
      const auto f = KOKKOS_LAMBDA (const MT& team) {
        const int ip = team.league_rank();
        const auto dl = get_diag(A, ip, 0);
        const auto d  = get_diag(A, ip, 1);
        const auto du = get_diag(A, ip, 2);
        const auto x  = get_x(X, ip);
        scream::tridiag::cr(team, dl, d, du, x);
      };
      Kokkos::parallel_for(policy, f);
    } else {
      if (in.oneA) {
        const auto f = KOKKOS_LAMBDA (const MT& team) {
          const int ip = team.league_rank();
          const auto dl = get_diag(A, ip, 0);
          const auto d  = get_diag(A, ip, 1);
          const auto du = get_diag(A, ip, 2);
          const auto x  = get_xs(X, ip);
          scream::tridiag::cr(team, dl, d, du, x);
        };
        Kokkos::parallel_for(policy, f);
      } else {
        const auto f = KOKKOS_LAMBDA (const MT& team) {
          const int ip = team.league_rank();
          const auto dl = get_diags(A, ip, 0);
          const auto d  = get_diags(A, ip, 1);
          const auto du = get_diags(A, ip, 2);
          const auto x  = get_xs(X, ip);
          assert(x.extent_int(1) == in.nrhs);
          assert(d.extent_int(1) == in.nrhs);
          scream::tridiag::cr(team, dl, d, du, x);
        };
        Kokkos::parallel_for(policy, f);
      }
    }
    Kokkos::fence();
    t1 = gettime();    
  } break;
  default:
    std::cout << "run does not support "
              << Solver::convert(in.method) << "\n";
  }

  const auto et = duration(t0, t1);
  printf("run: et %1.3e et/datum %1.3e\n", et, et/(in.nprob*in.nrow*in.nrhs));

  Real re; {
    auto Acopym = create_mirror_view(Acopy);
    auto Xm = create_mirror_view(X);
    auto Ym = create_mirror_view(Y);
    deep_copy(Acopym, Acopy);
    deep_copy(Xm, X);
    const auto ip = std::max(0, in.nprob-1);
    const auto dl = subview(Acopym, ip, 0, ALL(), ALL());
    const auto d  = subview(Acopym, ip, 1, ALL(), ALL());
    const auto du = subview(Acopym, ip, 2, ALL(), ALL());
    matvec(dl, d, du,
           subview(Xm, in.nprob-1, ALL(), ALL()),
           subview(Ym, in.nprob-1, ALL(), ALL()),
           nA, in.nrhs);
    re = reldif(subview(Bm, in.nprob-1, ALL(), ALL()),
                subview(Ym, in.nprob-1, ALL(), ALL()),
                in.nrhs);
  }
  if (re > 50*std::numeric_limits<Real>::epsilon())
    std::cout << "run: " << " re " << re << "\n";
}
} // namespace perf

int main (int argc, char **argv) {
  int num_failed = 0;
  scream::initialize_scream_session(argc, argv); {
    if (argc > 1) {
      // Performance test.
      perf::Input in;
      const auto stat = in.parse(argc, argv);
      if (stat) perf::run<scream::Real>(in);
    } else {
      // Correctness tests.
      num_failed = Catch::Session().run(argc, argv);
    }
  } scream::finalize_scream_session();
  return num_failed != 0 ? 1 : 0;
}
