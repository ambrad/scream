#include "atmosphere_dynamics.hpp"

// HOMMEXX Includes
#include "Context.hpp"
#include "FunctorsBuffersManager.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "ElementsForcing.hpp"
#include "GllFvRemap.hpp"

// Scream includes
#include "share/field/field_manager.hpp"
#include "dynamics/homme/homme_dimensions.hpp"

// Ekat includes
#include "ekat/ekat_assert.hpp"
#include "ekat/kokkos/ekat_subview_utils.hpp"
#include "ekat/ekat_pack.hpp"
#include "ekat/ekat_pack_kokkos.hpp"
#include "ekat/ekat_pack_utils.hpp"

extern "C" void gfr_init_hxx();

// Parse a name of the form "Physics PGN". Return -1 if not an FV physics grid
// name, otherwise N in pgN.
static int get_phys_grid_fv_param (const std::string& grid_name) {
  if (grid_name.size() < 11) return -1;
  if (grid_name.substr(0, 10) != "Physics PG") return -1;
  const auto param = grid_name.substr(10, std::string::npos);
  int N;
  std::istringstream ss(param);
  try {
    ss >> N;
  } catch (...) {
    N = -1;
  }
  return N;
}

namespace scream {

bool HommeDynamics::fv_phys_active () const {
  return m_phys_grid_pgN > 0;
}

void HommeDynamics::fv_phys_set_grids () {
  fprintf(stderr,"amb> fv_phys_set_grids\n");
  m_phys_grid_pgN = get_phys_grid_fv_param(m_phys_grid->name());
  fprintf(stderr,"amb> fv_phys_set_grids done %d\n", m_phys_grid_pgN);  
}

void HommeDynamics::fv_phys_requested_buffer_size_in_bytes () const {
  if (not fv_phys_active()) return;
  fprintf(stderr,"amb> fv_phys_requested_buffer_size_in_bytes\n");
  using namespace Homme;
  auto& c = Context::singleton();
  auto& gfr = c.create_if_not_there<GllFvRemap>();
  auto& fbm = c.create_if_not_there<FunctorsBuffersManager>();
  fbm.request_size(gfr.requested_buffer_size());
}

void HommeDynamics::fv_phys_initialize_impl () {
  if (not fv_phys_active()) return;
  fprintf(stderr,"amb> fv_phys_initialize_impl\n");
  using namespace Homme;
  auto& c = Context::singleton();
  auto& gfr = c.get<GllFvRemap>();
  gfr.reset(c.get<SimulationParams>());
  gfr_init_hxx();
}

// During restart, we can't overwrite T_mid, horiz_winds, and tracers, as these
// contain data used in homme_pre_process to compute tendencies.
struct HommeDynamics::GllFvRemapTmp {
  Homme::ExecView<Real***> T_mid;
  Homme::ExecView<Real****> horiz_winds, tracers;
};

// Copy physics T,uv state to FT,M to form tendencies in next dynamics step.
template <typename T_t, typename uv_t, typename FT_t, typename FM_t>
static void copy_prev (const int ncols, const int npacks,
                       const T_t& T, const uv_t& uv,
                       const FT_t& FT, const FM_t& FM) {
  using KT = KokkosTypes<DefaultDevice>;
  using ESU = ekat::ExeSpaceUtils<KT::ExeSpace>;
  const auto policy = ESU::get_default_team_policy(ncols, npacks);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const KT::MemberType& team) {
    const int& icol = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, npacks),
                         [&] (const int ilev) {
      FT(icol,ilev) = T(icol,ilev);
      FM(icol,0,ilev) = uv(icol,0,ilev);
      FM(icol,1,ilev) = uv(icol,1,ilev);
      FM(icol,2,ilev) = 0;
    });
  });
  Kokkos::fence();  
}

void HommeDynamics::fv_phys_dyn_to_fv_phys (const bool restart) {
  if (not fv_phys_active()) return;
  constexpr int N = HOMMEXX_PACK_SIZE;
  using Pack = ekat::Pack<Real,N>;
  const auto nlevs = m_phys_grid->get_num_vertical_levels();
  const auto npacks = ekat::PackInfo<N>::num_packs(nlevs);
  const auto& pgn = m_phys_grid->name();
  const auto ncols = m_phys_grid->get_num_local_dofs();
  const auto FT = m_helper_fields.at("FT_phys").get_view<Pack**>();
  const auto FM = m_helper_fields.at("FM_phys").get_view<Pack***>();
  if (restart) {
    constexpr int NGP = HOMMEXX_NP;
    const int nelem = m_dyn_grid->get_num_local_dofs()/(NGP*NGP);
    const auto npg = m_phys_grid_pgN*m_phys_grid_pgN;
    GllFvRemapTmp t;    
    t.T_mid = Homme::ExecView<Real***>("T_mid_tmp", nelem, npg, npacks*N);
    t.horiz_winds = Homme::ExecView<Real****>("horiz_winds_tmp", nelem, npg, 2, npacks*N);
    // Really need just the first tracer.
    const auto qsize = get_group_out("tracers", pgn).m_bundle->get_view<Real***>().extent_int(1);
    t.tracers = Homme::ExecView<Real****>("tracers_tmp", nelem, npg, qsize, npacks*N);
    remap_dyn_to_fv_phys(&t);
    assert(ncols == nelem*npg);
    Homme::ExecViewUnmanaged<Pack**> T(reinterpret_cast<Pack*>(t.T_mid.data()),
                                        ncols, npacks);
    Homme::ExecViewUnmanaged<Pack***> uv(reinterpret_cast<Pack*>(t.horiz_winds.data()),
                                         ncols, 2, npacks);
    copy_prev(ncols, npacks, T, uv, FT, FM);
  } else {
    remap_dyn_to_fv_phys();
    const auto T  = get_field_out("T_mid",pgn).get_view<const Pack**>();
    const auto uv = get_field_out("horiz_winds",pgn).get_view<const Pack***>();
    copy_prev(ncols, npacks, T, uv, FT, FM);
  }
  update_pressure(m_phys_grid);
}

// Q state: Map get_group_in("tracers", gn) to Homme's FQ.
// T,uv tendencies: Map F(T,M)_phys to Homme's F(T,M) tendency arrays.
void HommeDynamics::fv_phys_pre_process () {
  if (not fv_phys_active()) return;
  remap_fv_phys_to_dyn();
}

// Remap Homme state to get_field_out("x", m_phys_grid->name()) for x in (ps,
// phis, T_mid, omega, horiz_winds, tracers, pseudo_density). Then call
// update_pressure to update p_mid,int.
void HommeDynamics::fv_phys_post_process () {
  if (not fv_phys_active()) return;
  fv_phys_dyn_to_fv_phys();
}

void HommeDynamics::remap_dyn_to_fv_phys (GllFvRemapTmp* t) const {
  if (not fv_phys_active()) return;
  const auto& c = Homme::Context::singleton();
  auto& gfr = c.get<Homme::GllFvRemap>();
  const auto time_idx = c.get<Homme::TimeLevel>().n0;
  constexpr int NGP = HOMMEXX_NP;
  const int nelem = m_dyn_grid->get_num_local_dofs()/(NGP*NGP);
  const auto npg = m_phys_grid_pgN*m_phys_grid_pgN;
  const auto& gn = m_phys_grid->name();
  const auto nlev = get_field_out("T_mid", gn).get_view<Real**>().extent_int(1);
  const auto nq = get_group_out("tracers").m_bundle->get_view<Real***>().extent_int(1);
  assert(get_field_out("T_mid", gn).get_view<Real**>().extent_int(0) == nelem*npg);
  assert(get_field_out("horiz_winds", gn).get_view<Real***>().extent_int(1) == 2);
  
  const auto ps = Homme::GllFvRemap::Phys1T(
    get_field_out("ps", gn).get_view<Real*>().data(),
    nelem, npg);
  const auto phis = Homme::GllFvRemap::Phys1T(
    get_field_out("phis", gn).get_view<Real*>().data(),
    nelem, npg);
  const auto T = Homme::GllFvRemap::Phys2T(
    t ? t->T_mid.data() : get_field_out("T_mid", gn).get_view<Real**>().data(),
    nelem, npg, nlev);
  const auto omega = Homme::GllFvRemap::Phys2T(
    get_field_out("omega", gn).get_view<Real**>().data(),
    nelem, npg, nlev);
  const auto uv = Homme::GllFvRemap::Phys3T(
    t ? t->horiz_winds.data() : get_field_out("horiz_winds", gn).get_view<Real***>().data(),
    nelem, npg, 2, nlev);
  const auto q = Homme::GllFvRemap::Phys3T(
    t ? t->tracers.data() : get_group_out("tracers", gn).m_bundle->get_view<Real***>().data(),
    nelem, npg, nq, nlev);
  const auto dp = Homme::GllFvRemap::Phys2T(
    get_field_out("pseudo_density", gn).get_view<Real**>().data(),
    nelem, npg, nlev);

  gfr.run_dyn_to_fv_phys(time_idx, ps, phis, T, omega, uv, q, &dp);
  Kokkos::fence();
}

void HommeDynamics::remap_fv_phys_to_dyn () const {
  if (not fv_phys_active()) return;
  const auto& c = Homme::Context::singleton();
  auto& gfr = c.get<Homme::GllFvRemap>();
  const auto time_idx = c.get<Homme::TimeLevel>().n0;
  constexpr int NGP = HOMMEXX_NP;
  const int nelem = m_dyn_grid->get_num_local_dofs()/(NGP*NGP);
  const auto npg = m_phys_grid_pgN*m_phys_grid_pgN;
  const auto& gn = m_phys_grid->name();
  const auto nlev = m_helper_fields.at("FT_phys").get_view<const Real**>().extent_int(1);
  const auto nq = get_group_in("tracers", gn).m_bundle->get_view<const Real***>().extent_int(1);
  assert(m_helper_fields.at("FT_phys").get_view<const Real**>().extent_int(0) == nelem*npg);

  const auto uv_ndim = m_helper_fields.at("FM_phys").get_view<const Real***>().extent_int(1);
  assert(uv_ndim == 2 or uv_ndim == 3);

  const auto T = Homme::GllFvRemap::CPhys2T(
    m_helper_fields.at("FT_phys").get_view<const Real**>().data(),
    nelem, npg, nlev);
  const auto uv = Homme::GllFvRemap::CPhys3T(
    m_helper_fields.at("FM_phys").get_view<const Real***>().data(),
    nelem, npg, uv_ndim, nlev);
  const auto q = Homme::GllFvRemap::CPhys3T(
    get_group_in("tracers", gn).m_bundle->get_view<const Real***>().data(),
    nelem, npg, nq, nlev);
  
  gfr.run_fv_phys_to_dyn(time_idx, T, uv, q);
  Kokkos::fence();
  gfr.run_fv_phys_to_dyn_dss();
}

} // namespace scream
