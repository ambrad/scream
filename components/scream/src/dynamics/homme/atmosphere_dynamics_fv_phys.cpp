#include "atmosphere_dynamics.hpp"

// HOMMEXX Includes
#include "Context.hpp"
#include "FunctorsBuffersManager.hpp"
#include "SimulationParams.hpp"
#include "GllFvRemap.hpp"

// Scream includes

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
  m_phys_grid_pgN = get_phys_grid_fv_param(m_ref_grid->name());
}

void HommeDynamics::fv_phys_requested_buffer_size_in_bytes () const {
  if (not fv_phys_active()) return;
  using namespace Homme;
  auto& c = Context::singleton();
  auto& gfr = c.create_if_not_there<GllFvRemap>();
  gfr.reset(c.get<SimulationParams>());
  gfr_init_hxx();
  auto& fbm  = c.create_if_not_there<FunctorsBuffersManager>();
  fbm.request_size(gfr.requested_buffer_size());
}

void HommeDynamics::fv_phys_initialize_impl () {
  if (not fv_phys_active()) return;
}

void HommeDynamics::fv_phys_pre_process () {
  if (not fv_phys_active()) return;
}

void HommeDynamics::fv_phys_post_process () {
  if (not fv_phys_active()) return;
}

void HommeDynamics::fv_phys_restart_homme_state () {
  if (not fv_phys_active()) return;
}

void HommeDynamics::fv_phys_initialize_homme_state () {
  if (not fv_phys_active()) return;
}

} // namespace scream
