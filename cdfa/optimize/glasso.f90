!file: glasso.f90
subroutine glasso(Omega, Sigma, & !output
                  Sigma_pre, lambda, & !inputs :: data and parameter
                  ths, max_iter, & !inputs, optional :: opt. parameters
                  ths_lasso, max_iter_lasso, & !inputs, optional :: opt. param for lasso
                  dim_Sigma) !inputs :: dimensions
    implicit none
    
    !inputs :: dimensions
    integer, intent(in) :: dim_Sigma
    
    !inputs :: data and parameter
    real(8), dimension(dim_Sigma, dim_Sigma), intent(in) :: Sigma_pre, lambda
    
    !inputs :: opt. parameters
    integer, intent(in) :: max_iter
    real(8), intent(in) :: ths
    integer :: max_iter_lasso
    real(8) :: ths_lasso
    
!f2py integer optional, intent(in) :: max_iter = 100
!f2py real(8) optional, intent(in) :: ths = 0.001
!f2py integer optional, intent(in) :: max_iter_lasso = -1
!f2py real(8) optional, intent(in) :: ths_lasso = -1

    !in/outputs
    real(8), dimension(dim_Sigma, dim_Sigma) :: Omega, Sigma
    
!f2py intent(inplace) Omega
!f2py intent(inplace) Sigma

    ! local variables
    integer :: i, p, q, dim_dep
    real(8), dimension(dim_Sigma, dim_Sigma) :: Sigma_last
    
    if (max_iter_lasso < 0) then
        max_iter_lasso = max_iter
    end if
    if (ths_lasso < 0) then
        ths_lasso = ths
    end if

    do i = 1, max_iter
        Sigma_last = Sigma

        do p = 1, dim_Sigma
            dim_dep = &
                count((lambda(p,:) >= 0) .and. ((/(q,q=1,dim_Sigma)/) /= p))
            
            call glasso_p(Omega, Sigma, Sigma_pre, lambda, &
                          ths_lasso, max_iter_lasso, &
                          p, dim_Sigma, dim_dep)
        end do
                          
        if (maxval(abs(Sigma - Sigma_last)) < ths) then
            exit
        end if
    end do
end subroutine glasso