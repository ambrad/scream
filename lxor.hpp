#ifndef INCLUDE_LXOR
#define INCLUDE_LXOR

#include <mpi.h>
#include <memory>

// A reduction with MPI_LXOR doesn't seem to work on Summit. Make my own op
// instead.

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

static int all_reduce (MPI_Comm comm,
                       const LongLong* sendbuf, LongLong* rcvbuf, int count,
                       const Op& op) {
  return MPI_Allreduce(sendbuf, rcvbuf, count, MPI_LONG_LONG_INT, op.get(), comm);
}

static void lxor (void* invec, void* inoutvec, int* len, MPI_Datatype* datatype) {
  const int n = *len;
  const auto* s = reinterpret_cast<const LongLong*>(invec);
  auto* d = reinterpret_cast<LongLong*>(inoutvec);
  for (int i = 0; i < n; ++i) d[i] ^= s[i];
}

#endif
