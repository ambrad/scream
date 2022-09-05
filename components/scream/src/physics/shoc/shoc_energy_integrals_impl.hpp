#ifndef SHOC_ENERGY_INTEGRALS_IMPL_HPP
#define SHOC_ENERGY_INTEGRALS_IMPL_HPP

#include "shoc_functions.hpp" // for ETI only but harmless for GPU

namespace scream {
namespace shoc {

template<typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::shoc_energy_integrals(
  const MemberType&            team,
  const Int&                   nlev,
  const uview_1d<const Spack>& host_dse,
  const uview_1d<const Spack>& pdel,
  const uview_1d<const Spack>& rtm,
  const uview_1d<const Spack>& rcm,
  const uview_1d<const Spack>& u_wind,
  const uview_1d<const Spack>& v_wind,
  Scalar&                      se_int,
  Scalar&                      ke_int,
  Scalar&                      wv_int,
  Scalar&                      wl_int)
{
  using ExeSpaceUtils = ekat::ExeSpaceUtils<typename KT::ExeSpace>;
  const auto ggr = C::gravit;

#if 0
  // Compute se_int
  ExeSpaceUtils::view_reduction(team,0,nlev,
                                [&] (const int k) -> Spack {
    return host_dse(k)*pdel(k)/ggr;
  }, se_int);

  // Compute ke_int
  ExeSpaceUtils::view_reduction(team,0,nlev,
                                [&] (const int k) -> Spack {
    return sp(0.5)*(ekat::square(u_wind(k))+ekat::square(v_wind(k)))*pdel(k)/ggr;
  }, ke_int);

  // Compute wv_int
  ExeSpaceUtils::view_reduction(team,0,nlev,
                                [&] (const int k) -> Spack {
    return (rtm(k)-rcm(k))*pdel(k)/ggr;
  }, wv_int);

  // Compute wl_int
  ExeSpaceUtils::view_reduction(team,0,nlev,
                                [&] (const int k) -> Spack {
    return rcm(k)*pdel(k)/ggr;
  }, wl_int);
#else
  team.team_barrier();
  // Compute se_int
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,0,nlev),
                                [&] (const int k, Scalar& a) {
    a += host_dse(k)[0]*pdel(k)[0]/ggr;
  }, se_int);
  team.team_barrier();
  // Compute ke_int
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,0,nlev),
                                [&] (const int k, Scalar& a) {
    a += sp(0.5)*((u_wind(k)[0]*u_wind(k)[0])+(v_wind(k)[0]*v_wind(k)[0]))*pdel(k)[0]/ggr;
  }, ke_int);
  team.team_barrier();
  // Compute wv_int
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,0,nlev),
                                [&] (const int k, Scalar& a) {
    a += (rtm(k)[0]-rcm(k)[0])*pdel(k)[0]/ggr;
  }, wv_int);
  team.team_barrier();
  // Compute wl_int
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,0,nlev),
                                [&] (const int k, Scalar& a) {
    a += rcm(k)[0]*pdel(k)[0]/ggr;
  }, wl_int);
  team.team_barrier();
#endif
}

} // namespace shoc
} // namespace scream

#endif
