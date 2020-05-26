!file: glasso_p.f90
subroutine glasso_p(Omega, Sigma, & !output
                    Sigma_pre, lambda, & !inputs :: data and parameter
                    ths_lasso, max_iter_lasso, &
                    p, dim_Sigma, dim_dep) !inputs :: dimensions
    implicit none
    
    !inputs :: dimensions
    integer, intent(in) :: p, dim_Sigma, dim_dep
    
    !inputs :: data and parameter
    real(8), dimension(dim_Sigma, dim_Sigma), intent(in) :: Sigma_pre, lambda
    
    !inputs :: opt. parameters
    integer, intent(in) :: max_iter_lasso
    real(8), intent(in) :: ths_lasso
    
!f2py integer optional, intent(in) :: max_iter = 100
!f2py real(8) optional, intent(in) :: ths = 0.001

    !in/outputs
    real(8), dimension(dim_Sigma, dim_Sigma) :: Omega, Sigma
    
!f2py intent(inplace) Omega
!f2py intent(inplace) Sigma

    ! local variables
    integer :: q
    integer, dimension(dim_Sigma-1) :: notp
    integer, dimension(dim_dep) :: depp
    real(8), dimension(dim_dep) :: beta
    real(8), dimension(dim_Sigma, dim_Sigma) :: IOmega_notp
    
    notp = pack((/(q,q=1,dim_Sigma)/), (/(q,q=1,dim_Sigma)/)/=p)
    depp = pack((/(q,q=1,dim_Sigma)/), &
                (lambda(p,:)>=0) .and. ((/(q,q=1,dim_Sigma)/)/=p))
                 
    IOmega_notp = 0
    IOmega_notp(notp,notp) = &
        Sigma(notp,notp) - matmul(reshape(Sigma(notp,p), (/dim_Sigma-1,1/)), &
                                  reshape(Sigma(p,notp), (/1,dim_Sigma-1/)) / Sigma(p,p))
                                  
    Sigma(p,p) = Sigma_pre(p,p) + lambda(p,p)

    beta = Omega(p,depp)   
    call lasso(beta, Sigma(p,p) * IOmega_notp(depp,depp), - Sigma_pre(p,depp), &
               lambda(p,depp), ths_lasso, max_iter_lasso, dim_dep)

    Omega(p,:) = 0
    Omega(:,p) = 0
    Omega(p,depp) = beta
    Omega(depp,p) = beta

    Sigma(notp,p) = matmul(IOmega_notp(notp, depp), - beta * Sigma(p,p))
    Sigma(p,notp) = Sigma(notp, p)

    Omega(p,p) = (1 - dot_product(Omega(p, depp), Sigma(p, depp))) / Sigma(p,p)

    Sigma(notp,notp) = IOmega_notp(notp,notp) &
        + matmul(reshape(Sigma(notp,p), (/dim_Sigma-1,1/)), &
                 reshape(Sigma(p,notp), (/1,dim_Sigma-1/)) / Sigma(p,p))   

end subroutine glasso_p