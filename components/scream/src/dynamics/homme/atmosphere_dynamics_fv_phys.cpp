#include "atmosphere_dynamics.hpp"

// HOMMEXX Includes
#include "Context.hpp"
#include "FunctorsBuffersManager.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "ElementsForcing.hpp"
#include "GllFvRemap.hpp"

// Scream includes
#include "control/fvphyshack.hpp"
#include "share/field/field_manager.hpp"
#include "dynamics/homme/homme_dimensions.hpp"

// Ekat includes
#include "ekat/ekat_assert.hpp"
#include "ekat/kokkos/ekat_subview_utils.hpp"
#include "ekat/ekat_pack.hpp"
#include "ekat/ekat_pack_kokkos.hpp"

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
  assert(m_phys_grid_pgN < 0 || fvphyshack);
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
  fprintf(stderr,"amb> fv_phys_requested_buffer_size_in_bytes done\n");
}

void HommeDynamics::fv_phys_initialize_impl () {
  if (not fv_phys_active()) return;
  fprintf(stderr,"amb> fv_phys_initialize_impl\n");
  using namespace Homme;
  auto& c = Context::singleton();
  auto& gfr = c.get<GllFvRemap>();
  gfr.reset(c.get<SimulationParams>());
  fprintf(stderr,"amb> calling gfr_init_hxx\n");
  gfr_init_hxx();
  fprintf(stderr,"amb> gfr_init_hxx returned\n");
  fprintf(stderr,"amb> fv_phys_initialize_impl done\n");
}

void HommeDynamics::fv_phys_pre_process () {
  if (not fv_phys_active()) return;
  fprintf(stderr,"amb> fv_phys_pre_process\n");
  remap_fv_phys_to_dyn();
}

void HommeDynamics::fv_phys_post_process () {
  if (not fv_phys_active()) return;
  fprintf(stderr,"amb> fv_phys_post_process\n");
  remap_dyn_to_fv_phys();
  update_pressure(m_phys_grid);
}

void HommeDynamics::fv_phys_restart_homme_state () {
  if (not fv_phys_active()) return;
  fprintf(stderr,"amb> fv_phys_restart_homme_state\n");
}

void HommeDynamics::fv_phys_initialize_homme_state () {
  if (not fv_phys_active()) return;
  fprintf(stderr,"amb> fv_phys_initialize_homme_state\n");
}

void HommeDynamics::remap_dyn_to_fv_phys () const {
  if (not fv_phys_active()) return;
  fprintf(stderr,"amb> remap_dyn_to_fv_phys\n");
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
    get_field_out("T_mid", gn).get_view<Real**>().data(),
    nelem, npg, nlev);
  const auto omega = Homme::GllFvRemap::Phys2T(
    get_field_out("omega", gn).get_view<Real**>().data(),
    nelem, npg, nlev);
  const auto uv = Homme::GllFvRemap::Phys3T(
    get_field_out("horiz_winds", gn).get_view<Real***>().data(),
    nelem, npg, 2, nlev);
  const auto q = Homme::GllFvRemap::Phys3T(
    get_group_out("tracers", gn).m_bundle->get_view<Real***>().data(),
    nelem, npg, nq, nlev);
  const auto dp = Homme::GllFvRemap::Phys2T(
    get_field_out("pseudo_density", gn).get_view<Real**>().data(),
    nelem, npg, nlev);

  gfr.run_dyn_to_fv_phys(time_idx, ps, phis, T, omega, uv, q, &dp);
  fprintf(stderr,"amb> remap_dyn_to_fv_phys done\n");
}

void HommeDynamics::remap_fv_phys_to_dyn () const {
  if (not fv_phys_active()) return;
  fprintf(stderr,"amb> remap_fv_phys_to_dyn\n");
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
  // FM in Homme includes a component for omega, but the physics don't modify
  // omega. Thus, zero FM so that the third component is 0.
  const auto& forcing = c.get<Homme::ElementsForcing>();
  Kokkos::deep_copy(forcing.m_fm, 0);

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
  gfr.run_fv_phys_to_dyn_dss();
  fprintf(stderr,"amb> remap_fv_phys_to_dyn done\n");
}

} // namespace scream
