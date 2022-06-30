#include "atmosphere_dynamics.hpp"

// HOMMEXX Includes
#include "Context.hpp"
#include "FunctorsBuffersManager.hpp"
#include "SimulationParams.hpp"
#include "GllFvRemap.hpp"

// Scream includes
#include "control/fvphyshack.hpp"

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

void HommeDynamics
::fv_phys_set_grids (const std::shared_ptr<const GridsManager>& grids_manager) {
  fprintf(stderr,"amb> fv_phys_set_grids\n");
  const auto grids = grids_manager->supported_grids();
  for (const auto& grid : grids) {
    fprintf(stderr,"amb> fv_phys_set_grids process %s\n",grid.c_str());
    m_phys_grid_pgN = get_phys_grid_fv_param(grid);
    assert(m_phys_grid_pgN < 0 || fvphyshack);
    if (m_phys_grid_pgN > 0) break;
  }
  fprintf(stderr,"amb> fv_phys_set_grids done %d\n", m_phys_grid_pgN);
}

void HommeDynamics::fv_phys_requested_buffer_size_in_bytes () const {
  if (not fv_phys_active()) return;
  fprintf(stderr,"amb> fv_phys_requested_buffer_size_in_bytes\n");
  using namespace Homme;
  auto& c = Context::singleton();
  auto& gfr = c.create_if_not_there<GllFvRemap>();
  auto& fbm  = c.create_if_not_there<FunctorsBuffersManager>();
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
}

void HommeDynamics::fv_phys_post_process () {
  if (not fv_phys_active()) return;
  fprintf(stderr,"amb> fv_phys_post_process\n");
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
}

void HommeDynamics::remap_fv_phys_to_dyn () const {
  if (not fv_phys_active()) return;
  fprintf(stderr,"amb> remap_fv_phys_to_dynys\n");
}

} // namespace scream
