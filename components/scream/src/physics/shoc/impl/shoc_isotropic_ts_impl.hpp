#ifndef SHOC_ISOTROPIC_TS_IMPL_HPP
#define SHOC_ISOTROPIC_TS_IMPL_HPP

#include "shoc_functions.hpp" // for ETI only but harmless for GPU

namespace scream {
namespace shoc {

/*
 * Implementation of shoc isotropic_ts. Clients should NOT
 * #include this file, but include shoc_functions.hpp instead.
 */

template<typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::isotropic_ts(
  const MemberType&            team,
  const Int&                   nlev,
  const Scalar&                brunt_int,
  const uview_1d<const Spack>& tke,
  const uview_1d<const Spack>& a_diss,
  const uview_1d<const Spack>& brunt,
  const uview_1d<Spack>&       isotropy)
{

  //constants from physics/share
  const Scalar ggr = C::gravit;

  //Declare constants
  const Scalar lambda_low   = 0.001;
  const Scalar lambda_high  = 0.04;
  const Scalar lambda_slope = 2.65;
  const Scalar lambda_thresh= 0.02;
  const Scalar maxiso       = 20000; // Return to isotropic timescale [s]

  const Int nlev_pack = ekat::npack<Spack>(nlev);

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev_pack), [&] (const Int& k) {
    // define the time scale
    const Scalar tscale = (2*tke(k)/a_diss(k))[0];

    // define a damping term "lambda" based on column stability
    Scalar lambda = lambda_low + ((brunt_int/ggr)-lambda_thresh)*lambda_slope;
    if (lambda > lambda_high) lambda = lambda_high;
    if (lambda < lambda_low ) lambda = lambda_low ;

    const Scalar buoy_sgs_save = brunt(k)[0];
    if (buoy_sgs_save <= 0) lambda = 0;
    lambda = 0;

    // Compute the return to isotropic timescale
    auto tmp = tscale/(1 + lambda*buoy_sgs_save*tscale*tscale);
    if (tmp > maxiso) tmp = maxiso;
    isotropy(k)[0] = tmp;
  });
}

} // namespace shoc
} // namespace scream

#endif
