/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"
#include "Context.hpp"
#include "VerticalRemapManager.hpp"
#include "RemapFunctor.hpp"

namespace Homme {
using cti = ComposeTransportImpl;

void ComposeTransportImpl
::remap_v (const ExecViewUnmanaged<const Scalar*[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>& dp3d,
           const int np1, const ExecViewUnmanaged<const Scalar*[NP][NP][NUM_LEV]>& dp,
           const ExecViewUnmanaged<Scalar*[2][NP][NP][NUM_LEV]>& v) {
  using Kokkos::parallel_for;
  const auto& vrm = Context::singleton().get<VerticalRemapManager>();
  const auto r = vrm.get_remapper();
  const auto policy = Kokkos::RangePolicy<ExecSpace>(0, dp3d.extent_int(0)*NP*NP*NUM_LEV*2);
  const auto pre = KOKKOS_LAMBDA (const int idx) {
    int ie, q, i, j, k;
    cti::idx_ie_q_ij_nlev<NUM_LEV>(2, idx, ie, q, i, j, k);
    v(ie,q,i,j,k) *= dp3d(ie,np1,i,j,k);
  };
  parallel_for(policy, pre);
  Kokkos::fence();
  r->remap1(dp3d, np1, dp, v, 2);
  Kokkos::fence();
  const auto post = KOKKOS_LAMBDA (const int idx) {
    int ie, q, i, j, k;
    cti::idx_ie_q_ij_nlev<NUM_LEV>(2, idx, ie, q, i, j, k);
    v(ie,q,i,j,k) /= dp(ie,i,j,k);
  };
  parallel_for(policy, post);
}

void ComposeTransportImpl::remap_q (const TimeLevel& tl) {
  GPTLstart("compose_vertical_remap");
  const auto np1 = tl.np1;
  const auto np1_qdp = tl.np1_qdp;
  const auto dp = m_derived.m_divdp;
  const auto dp3d = m_state.m_dp3d;
  const auto qdp = m_tracers.qdp;
  const auto q = m_tracers.Q;
  const int nq = m_tracers.num_tracers();
  {
    // this checks
    const auto dp3dh = Kokkos::create_mirror_view(dp3d);
    Kokkos::deep_copy(dp3dh, dp3d);
    const auto qdph = Kokkos::create_mirror_view(qdp);
    Kokkos::deep_copy(qdph, qdp);
    Real qmin[32];
    for (int qi = 0; qi < nq; ++qi) qmin[qi] = 1e30;
    for (int i = 0; i < q.extent_int(0); ++i)
      for (int j = 0; j < q.extent_int(1); ++j)
        for (int k = 0; k < q.extent_int(2); ++k)
          for (int l = 0; l < q.extent_int(3); ++l)
            for (int m = 0; m < q.extent_int(4); ++m)
              qmin[j] = std::min(qmin[j], qdph(i,np1_qdp,j,k,l,m)[0]/dp3dh(i,np1,k,l,m)[0]);
    for (int i = 0; i < nq; ++i)
      if (qmin[i] < 0)
        fprintf(stderr,"amb> vr before %d %1.3e\n",i,qmin[i]);
    // this also checks
    Real dpmin = 1e30;
    const auto dph = Kokkos::create_mirror_view(dp);
    Kokkos::deep_copy(dph, dp);
    for (int i = 0; i < dph.extent_int(0); ++i)
      for (int j = 0; j < dph.extent_int(1); ++j)
        for (int k = 0; k < dph.extent_int(2); ++k)
          for (int l = 0; l < dph.extent_int(3); ++l)
            dpmin = std::min(dpmin, dph(i,j,k,l)[0]);
    if (dpmin <= 0)
      fprintf(stderr,"amb> vr before dpmin %1.3e\n",dpmin);
  }
  const auto& vrm = Context::singleton().get<VerticalRemapManager>();
  const auto r = vrm.get_remapper();
  amb_dbg = true;
  r->remap1(dp, dp3d, np1, qdp, np1_qdp, nq);
  amb_dbg = false;
  const auto post = KOKKOS_LAMBDA (const int idx) {
    int ie, iq, i, j, k;
    cti::idx_ie_q_ij_nlev<NUM_LEV>(nq, idx, ie, iq, i, j, k);
    q(ie,iq,i,j,k) = qdp(ie,np1_qdp,iq,i,j,k)/dp3d(ie,np1,i,j,k);
  };
  const auto policy = Kokkos::RangePolicy<ExecSpace>(0, dp3d.extent_int(0)*NP*NP*NUM_LEV*nq);
  Kokkos::fence();
  Kokkos::parallel_for(policy, post);
  Kokkos::fence();
  GPTLstop("compose_vertical_remap");
  {
    // this fails
    const auto qh = Kokkos::create_mirror_view(q);
    Kokkos::deep_copy(qh, q);
    Real qmin[32] = {0};
    for (int i = 0; i < qh.extent_int(0); ++i)
      for (int j = 0; j < qh.extent_int(1); ++j)
        for (int k = 0; k < qh.extent_int(2); ++k)
          for (int l = 0; l < qh.extent_int(3); ++l)
            for (int m = 0; m < qh.extent_int(4); ++m)
              qmin[j] = std::min(qmin[j], qh(i,j,k,l,m)[0]);
    for (int i = 0; i < nq; ++i)
      if (qmin[i] < 0)
        fprintf(stderr,"amb> vr after %d %1.3e\n",i,qmin[i]);
  }
}

} // namespace Homme
