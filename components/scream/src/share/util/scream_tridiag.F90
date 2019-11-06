#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include "scream_config.f"
#ifdef SCREAM_DOUBLE_PRECISION
# define c_real c_double
#else
# define c_real c_float
#endif

! These routines are meant only for BFB testing. It solves a diagonally
! domainant tridiagonal system A x = b, with (dl,d,du) the tridiags and x = b on
! input. See scream_tridag.hpp for performant solvers.

module scream_tridag
  use iso_c_binding, only: c_int, c_real

  implicit none

contains

  subroutine tridiag_diagdom_bfb_a1x1(n, dl, d, du, x) bind(c)
    integer(c_int), value, intent(in) :: n
    real(c_real), intent(inout) :: dl(n), d(n), du(n), x(n)
    
    real(c_real) :: dli
    integer :: i

    do i = 2,n
       dli = dl(i)/d(i-1)
       d(i) = d(i) - dli*du(i-1)
       x(i) = x(i) - dli*x (i-1)
    end do
    x(n) = x(n)/d(n)
    do i = n,2,-1
       x(i-1) = (x(i-1) - du(i-1)*x(i))/d(i-1)
    end do
  end subroutine tridiag_diagdom_bfb_a1x1

  subroutine tridiag_diagdom_bfb_a1xm(n, nrhs, dl, d, du, x) bind(c)
    integer(c_int), value, intent(in) :: n, nrhs
    real(c_real), intent(inout) :: dl(n), d(n), du(n), x(nrhs,n)
    
    real(c_real) :: dli
    integer :: i, j

    do i = 2,n
       dli = dl(i)/d(i-1)
       d(i) = d(i) - dli*du(i-1)
       do j = 1,nrhs
          x(j,i) = x(j,i) - dli*x(j,i-1)
       end do
    end do
    do j = 1,nrhs
       x(j,n) = x(j,n)/d(n)
    end do
    do i = n,2,-1
       do j = 1,nrhs
          x(j,i-1) = (x(j,i-1) - du(i-1)*x(j,i))/d(i-1)
       end do
    end do
  end subroutine tridiag_diagdom_bfb_a1xm
  
end module scream_tridag
