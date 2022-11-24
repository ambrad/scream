#ifndef SHOC_TKE_IMPL_HPP
#define SHOC_TKE_IMPL_HPP

#include "shoc_functions.hpp" // for ETI only but harmless for GPU

namespace scream {
namespace shoc {

/*
 * Implementation of shoc shoc_tke. Clients should NOT
 * #include this file, but include shoc_functions.hpp instead.
 */

/*
 * Thus function advances the SGS
 * TKE equation due to shear production, buoyant
 * production, and dissipation processes.
 */

template<typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>::shoc_tke(
  const MemberType&            team,
  const Int&                   nlev,
  const Int&                   nlevi,
  const Scalar&                dtime,
  const uview_1d<const Spack>& wthv_sec,
  const uview_1d<const Spack>& shoc_mix,
  const uview_1d<const Spack>& dz_zi,
  const uview_1d<const Spack>& dz_zt,
  const uview_1d<const Spack>& pres,
  const uview_1d<const Spack>& u_wind,
  const uview_1d<const Spack>& v_wind,
  const uview_1d<const Spack>& brunt,
  const Scalar&                obklen,
  const uview_1d<const Spack>& zt_grid,
  const uview_1d<const Spack>& zi_grid,
  const Scalar&                pblh,
  const Workspace&             workspace,
  const uview_1d<Spack>&       tke,
  const uview_1d<Spack>&       tk,
  const uview_1d<Spack>&       tkh,
  const uview_1d<Spack>&       isotropy,
  Result& r)
{
  // Define temporary variables
  uview_1d<Spack> sterm_zt, a_diss, sterm;
  workspace.template take_many_contiguous_unsafe<3>(
    {"sterm_zt", "a_diss", "sterm"},
    {&sterm_zt, &a_diss, &sterm});

  // Compute integrated column stability in lower troposphere
  team.team_barrier(); //amb
  Kokkos::memory_fence(); //amb
  Scalar brunt_int(0);
  integ_column_stability(team,nlev,dz_zt,pres,brunt,brunt_int);
  team.team_barrier(); //amb
  Kokkos::memory_fence(); //amb
  const Int nlev_pack = ekat::npack<Spack>(nlev);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev_pack), [&] (const Int& k) {
    combine(r.v[84], brunt_int);
  });
  team.team_barrier(); //amb
  // Compute shear production term, which is on interface levels
  // This follows the methods of Bretheron and Park (2010)
  compute_shr_prod(team,nlevi,nlev,dz_zi,u_wind,v_wind,sterm);
  Kokkos::memory_fence(); //amb
  team.team_barrier(); //amb
  // Interpolate shear term from interface to thermo grid
  linear_interp(team,zi_grid,zt_grid,sterm,sterm_zt,nlevi,nlev,0);
  Kokkos::memory_fence(); //amb
  team.team_barrier(); //amb
  Kokkos::single(
    Kokkos::PerTeam(team),
    [&] () {
      for (int k = 0; k < nlev; ++k) {
        combine(r.v[73], sterm_zt[k]);
        combine(r.v[74], tke[k]);
        combine(r.v[75], wthv_sec[k]); // diff
        combine(r.v[76], shoc_mix[k]);
        combine(r.v[77], tk[k]);
      }
    });
  team.team_barrier(); //amb
  // Advance sgs TKE
  adv_sgs_tke(team,nlev,dtime,shoc_mix,wthv_sec,sterm_zt,tk,tke,a_diss);
  Kokkos::memory_fence(); //amb
  team.team_barrier(); //amb
  Kokkos::single(
    Kokkos::PerTeam(team),
    [&] () {
      for (int k = 0; k < nlev; ++k) {
        combine(r.v[78], a_diss[k]);
        combine(r.v[79], tke[k]);
        combine(r.v[80], brunt[k]);
      }
      combine(r.v[81], obklen);
      combine(r.v[82], pblh);
    });
  team.team_barrier(); //amb
  // Compute isotropic time scale [s]
  isotropic_ts(team,nlev,brunt_int,tke,a_diss,brunt,isotropy);
  Kokkos::memory_fence(); //amb
  team.team_barrier(); //amb
  Kokkos::single(
    Kokkos::PerTeam(team),
    [&] () {
      for (int k = 0; k < nlev; ++k)
        combine(r.v[83], isotropy[k]);
    });
  team.team_barrier(); //amb
  // Compute eddy diffusivity for heat and momentum
  eddy_diffusivities(team,nlev,obklen,pblh,zt_grid,shoc_mix,sterm_zt,isotropy,tke,tkh,tk);
  Kokkos::memory_fence(); //amb
  team.team_barrier(); //amb
  // Release temporary variables from the workspace
  workspace.template release_many_contiguous<3>(
    {&sterm_zt, &a_diss, &sterm});
}

} // namespace shoc
} // namespace scream

#endif
