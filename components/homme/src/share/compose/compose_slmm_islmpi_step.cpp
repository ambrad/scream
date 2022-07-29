#include "compose_slmm_islmpi.hpp"

struct Op {
  typedef std::shared_ptr<Op> Ptr;

  Op (MPI_User_function* function, bool commute) {
    MPI_Op_create(function, static_cast<int>(commute), &op_);
  }

  ~Op () { MPI_Op_free(&op_); }

  const MPI_Op& get () const { return op_; }

private:
  MPI_Op op_;
};

typedef long long LongLong;

int all_reduce (const homme::mpi::Parallel& p,
                const LongLong* sendbuf, LongLong* rcvbuf, int count,
                const Op& op) {
  return MPI_Allreduce(sendbuf, rcvbuf, count, MPI_LONG_LONG_INT, op.get(),
                       p.comm());
}

static void
lxor (void* invec, void* inoutvec, int* len, MPI_Datatype* datatype) {
  const int n = *len;
  const auto* s = reinterpret_cast<const LongLong*>(invec);
  auto* d = reinterpret_cast<LongLong*>(inoutvec);
  for (int i = 0; i < n; ++i)
    d[i] ^= s[i];
}

namespace homme {
namespace islmpi {

// dep_points is const in principle, but if lev <=
// semi_lagrange_nearest_point_lev, a departure point may be altered if the
// winds take it outside of the comm halo.
template <typename MT>
void step (
  IslMpi<MT>& cm, const Int nets, const Int nete,
  Real* dep_points_r,           // dep_points(1:3, 1:np, 1:np)
  Real* q_min_r, Real* q_max_r) // q_{min,max}(lev, 1:np, 1:np, 1:qsize, ie-nets+1)
{
  {
    const auto& t = *cm.tracer_arrays;
    const auto q = Kokkos::create_mirror_view(cm.tracer_arrays->q);
    Kokkos::deep_copy(q, t.q);
    LongLong vl[32] = {0}, vg[32] = {0};
    for (int ci = 0; ci < t.nelemd; ++ci)
      for (int iq = 0; iq < t.qsize; ++iq)
        for (int k = 0; k < t.np2; ++k)
          for (int lev = 0; lev < t.nlev; ++lev)
            vl[iq] ^= *reinterpret_cast<const LongLong*>(&q(ci, iq, k, lev));
    Op op(lxor, true);
    all_reduce(*cm.p, vl, vg, t.qsize, op);
    if (cm.p->amroot())
      for (int iq = 0; iq < t.qsize; ++iq)
        fprintf(stderr, "amb q> %d %lld\n", iq, vg[iq]);
  }
  
  using slmm::Timer;

  slmm_assert(cm.np == 4);
#ifdef COMPOSE_PORT
  slmm_assert(nets == 0 && nete+1 == cm.nelemd);
#endif

#ifdef COMPOSE_PORT
  const auto& dep_points = cm.tracer_arrays->dep_points;
  const auto& q_min = cm.tracer_arrays->q_min;
  const auto& q_max = cm.tracer_arrays->q_max;
#else
  const DepPointsH<MT> dep_points(dep_points_r, cm.nelemd, cm.nlev, cm.np2);
  const QExtremaH<MT>
    q_min(q_min_r, cm.nelemd, cm.qsize, cm.nlev, cm.np2),
    q_max(q_max_r, cm.nelemd, cm.qsize, cm.nlev, cm.np2);
#endif

  // Partition my elements that communicate with remotes among threads, if I
  // haven't done that yet.
  { Timer t("01_mylid");
    if (cm.mylid_with_comm_tid_ptr_h.capacity() == 0)
      init_mylid_with_comm_threaded(cm, nets, nete); }
  // Set up to receive departure point requests from remotes.
  { Timer t("02_setup_irecv");
    setup_irecv(cm); }
  // Determine where my departure points are, and set up requests to remotes as
  // well as to myself to fulfill these.
  { Timer t("03_adp");
    analyze_dep_points(cm, nets, nete, dep_points); }
  { Timer t("04_pack_pass1");
    pack_dep_points_sendbuf_pass1(cm); }
  { Timer t("05_pack_pass2");
    pack_dep_points_sendbuf_pass2(cm, dep_points); }
  // Send requests.
  { Timer t("06_isend");
    isend(cm); }
  // While waiting, compute q extrema in each of my elements.
  { Timer t("07_q_extrema");
    calc_q_extrema(cm, nets, nete); }
  // Wait for the departure point requests. Since this requires a thread
  // barrier, at the same time make sure the send buffer is free for use.
  { Timer t("08_recv_and_wait");
    recv_and_wait_on_send(cm); }
  // Compute the requested q for departure points from remotes.
  calc_rmt_q(cm);
  // Send q data.
  { Timer t("10_isend");
    isend(cm, true /* want_req */, true /* skip_if_empty */); }
  // Set up to receive q for each of my departure point requests sent to
  // remotes. We can't do this until the OpenMP barrier in isend assures that
  // all threads are done with the receive buffer's departure points.
  { Timer t("11_setup_irecv");
    setup_irecv(cm, true /* skip_if_empty */); }
  // While waiting to get my data from remotes, compute q for departure points
  // that have remained in my elements.
  { Timer t("12_own_q");
    calc_own_q(cm, nets, nete, dep_points, q_min, q_max); }
  // Receive remote q data and use this to fill in the rest of my fields.
  { Timer t("13_recv");
    recv(cm, true /* skip_if_empty */); }
  { Timer t("14_copy_q");
    copy_q(cm, nets, q_min, q_max); }
  // Wait on send buffer so it's free to be used by others.
  { Timer t("15_wait_on_send");
    wait_on_send(cm, true /* skip_if_empty */); }
}

template void step(IslMpi<ko::MachineTraits>&, const Int, const Int, Real*, Real*, Real*);

} // namespace islmpi
} // namespace homme
