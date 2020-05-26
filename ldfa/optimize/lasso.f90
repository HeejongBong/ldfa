!file: lasso.f90
subroutine lasso(beta, & !output
                 V, u, lambda, & !inputs :: data and parameter
                 ths, max_iter, & !inputs, optional :: opt. parameters
                 dim_beta) !inputs :: dimensions
    implicit none
    
    !inputs :: dimensions
    integer, intent(in) :: dim_beta
    
    !inputs :: data and parameter
    real(8), dimension(dim_beta, dim_beta), intent(in) :: V
    real(8), dimension(dim_beta), intent(in) :: u, lambda
    
    !inputs :: opt. parameters
    integer, intent(in) :: max_iter
    real(8), intent(in) :: ths
    
!f2py integer optional, intent(in) :: max_iter = 100
!f2py real(8) optional, intent(in) :: ths = 0.001

    !in/outputs
    real(8), dimension(dim_beta) :: beta
    
!f2py intent(inplace) beta

    ! local variables
    integer :: i, q
    real(8), dimension(dim_beta) :: beta_last
    
    do i = 1, max_iter
        beta_last = beta
        
        do q = 1, dim_beta
            beta(q) = u(q) - dot_product(V(q,:q-1), beta(:q-1)) &
                      - dot_product(V(q,q+1:), beta(q+1:))
            beta(q) = sign(max(abs(beta(q))-lambda(q), 0.0) / V(q,q), &
                           beta(q))
        end do
        
        if (maxval(abs(beta - beta_last)) < ths) then
            exit
        end if
    end do
            
end subroutine lasso