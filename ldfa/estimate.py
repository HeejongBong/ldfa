# -*- coding: utf-8 -*-
"""LDFA-H: Latent Dynamic Factor Analysis of High-dimensional time-series

This module implements the fitting algorithm of LDFA-H and the accessory functions to
facilitate the associate analyses or inferences. 

Todo
----
    * Correct function ``fit_Phi``.

.. _[1] Bong et al. (2020). Latent Dynamic Factor Analysis of High-Dimensional Neural Recordings. Submitted to NeurIPS2020.

"""

import time, sys, traceback
import numpy as np
from scipy import linalg

import ldfa.optimize as core

def _generate_lambda_glasso(bin_num, lambda_glasso, offset, lambda_diag=None):
    """Generate sparsity penalty matrix Lambda for a submatrix in Pi."""
    lambda_glasso_out = np.full((bin_num, bin_num), -1) + (1+lambda_glasso) * \
           (np.abs(np.arange(bin_num) - np.arange(bin_num)[:,np.newaxis]) <= offset)
    if lambda_diag:
        lambda_glasso_out[np.arange(bin_num), np.arange(bin_num)] = lambda_diag
    return lambda_glasso_out

def _switch_back(Sigma, Phi_S, Gamma_T, Phi_T, beta):
    """Make the initial Phi_S positive definite."""
    w, v = np.linalg.eig(Sigma)
    sqrtS = (v*np.sqrt(w)[...,None,:])@v.transpose(0,2,1)
    alpha = np.min(np.linalg.eigvals(
        sqrtS @ linalg.block_diag(*Gamma_T) @ sqrtS),-1)/2
    
    return (Sigma - alpha[:,None,None]*linalg.block_diag(*Phi_T), 
            [P + (b*alpha)@b.T for P, b in zip(Phi_S, beta)])

def _make_PD(m, thres_eigen=1e-4):
    """Make a coariance PSD based on its eigen decomposition."""
    s, v = np.linalg.eigh(m)
    if np.min(s)<=thres_eigen*np.max(s):
        delta = thres_eigen*np.max(s) - np.min(s)
        s = s + delta
    return (v@np.diag(s)@np.linalg.inv(v))

def _temporal_est(V_eps_T, ar_order):
    """Perform temporal estimate given V_T.
    
    .. _[1] Bickel, P. J. and Levina, E. (2008). Regularized estimation of large covariance matrices. Ann. Statist., 36(1):199–227.
    """
    num_time = V_eps_T.shape[0]
    resids = np.zeros(num_time)
    Amatrix = np.zeros([num_time, num_time])
    
    resids[0] = V_eps_T[0,0]
    for i in np.arange(1, ar_order):
        Amatrix[i,:i] = np.linalg.pinv(V_eps_T[:i,:i]) @ V_eps_T[:i,i]
        resids[i] = V_eps_T[i,i] \
            - V_eps_T[i,:i] @ np.linalg.pinv(V_eps_T[:i,:i]) @ V_eps_T[:i,i]
        
    for i in np.arange(ar_order, num_time):
        Amatrix[i,i-ar_order:i] = np.linalg.pinv(V_eps_T[i-ar_order:i,i-ar_order:i]) \
            @ V_eps_T[i-ar_order:i,i]
        resids[i] = V_eps_T[i,i] \
            - V_eps_T[i,i-ar_order:i] \
            @ np.linalg.pinv(V_eps_T[i-ar_order:i,i-ar_order:i]) \
            @ V_eps_T[i-ar_order:i,i]
    
#     invIA = np.linalg.pinv(np.eye(num_time) - Amatrix)
#     Psi_T_hat = invIA @ np.diag(resids) @ invIA.T
#     Gamma_T_hat = np.linalg.inv(Psi_T_hat)
    
    Gamma_T_hat = (np.eye(num_time)-Amatrix).T @ np.diag(1/resids) \
                  @ (np.eye(num_time)-Amatrix)
    Psi_T_hat = np.linalg.pinv(Gamma_T_hat)
    
    return Gamma_T_hat, Psi_T_hat

def fit(data, num_f, lambda_cross, offset_cross, 
        lambda_auto=None, offset_auto=None, lambda_aug=0,
        ths_ldfa=1e-2, max_ldfa=1000, ths_glasso=1e-8, max_glasso=1000,
        ths_lasso=1e-8, max_lasso=1000, params_init=dict(), make_PD=False,
        verbose=False):
    """The main function to perform multi-factor LDFA-H estimation.
    
    Parameters
    ----------
    data: list of (N, p_k, T) ndarrays
        Observed data from K areas. Data from each area k consists of p_k-variate
        time-series over T time bins in N trials. 
    num_f: int
        The number of factors.
    lambda_cross, lambda_auto: float
        The sparsity penalty parameter for the inverse cross-correlation and inverse
        auto-correlation matrix, respectively. The default value for lambda_auto is 0.
    offset_cross, offset_auto: int
        The bandwidth parameter for the inverse cross-correlation matrix and inverse
        auto-correlation matrix, respectively. The default value for offset_auto is the
        given value of offset_cross.
    ths_ldfa, ths_glasso, ths_lasso: float, optional
        The threshold values for deciding the convergence of the main iteration, the
        glasso iteration, and the lasso iteration, respectively.
    max_ldfa, max_glasso, max_lasso: int, optional
        The maximum number of iteration for the main iteration, the glasso iteration,
        and the lasso iteration, respectively.
    beta_init: list of (p_k, num_f) ndarrays, optional
        Custom initial values for beta. If not given, beta is initialized by CCA.
    make_PD: boolean, optional
        Switch for manual positive definitization. If data does not generate a positive
        definite estimate of the covariance matrix, ``make_PD = True`` helps with 
        maintaining the matrix positive definite throughout the fitting algorithm. The
        default value is False for the sake of running time.
    verbose: boolean, optional
        Swith for vocal feedback throughout the fitting algorithm. The default value is
        False.
   
    Returns
    -------
    Pi: (K*T, K*T) ndarray
        The estimated sparse inverse correlation matrix.
    Rho: (K*T, K*T) ndarray
        The estimated correlation matrix before sparsification. Note that Rho != Pi^{-1}.
    params: dict
        The dictionary of the estimated parameters. It provides with the estimation of
            Omega: (num_f, K*T, K*T) ndarray; 
            Gamma_S: a list of (p_k, p_k) ndarrays for k = 1, ..., K;
            Gamma_T: a list of (T, T) ndarrays for k = 1, ..., K;
            beta: a list of (p_k, num_f) ndarrays for k = 1, ..., K; and
            mu: a list of (p_k, T) ndarrays for k = 1, ..., K. 
    
    Examples
    --------
    Pi, Rho, params =\
        fit(data, num_f, lambda_cross, offset_cross, lambda_auto, offset_auto)
                 
    .. _[1] Bong et al. (2020). Latent Dynamic Factor Analysis of High-Dimensional Neural Recordings. Submitted to NeurIPS2020.
    
    """
    dims = [dat.shape[1] for dat in data]
    num_time = data[0].shape[2]
    num_trial = data[0].shape[0]

    # get full_graph
    if lambda_auto is None:
        lambda_auto = lambda_cross
    if offset_auto is None:
        offset_auto = offset_cross
    lambda_glasso_auto = _generate_lambda_glasso(num_time, lambda_auto, 
                                                 offset_auto)
    lambda_glasso_cross = _generate_lambda_glasso(num_time, lambda_cross,
                                                  offset_cross)
    lambda_glasso = np.array(np.block(
        [[lambda_glasso_auto if j==i else lambda_glasso_cross
          for j, _ in enumerate(data)]
         for i, _ in enumerate(data)]))
    
    # set mu
    mu= [np.mean(dat, 0) for dat in data]
    
    # initialization
    if all(key in params_init for key in ('Omega', 'mu', 'beta', 'Gamma_S', 'Gamma_T')):
        Omega = params_init['Omega']; mu = params_init['mu']; beta = params_init['beta']
        Gamma_S = params_init['Gamma_S']; Gamma_T = params_init['Gamma_T']
        
        Sigma = np.linalg.inv(Omega)
        sig = np.sqrt(np.diagonal(Sigma,0,1,2))
        Rho = Sigma/sig[:,None,:]/sig[:,:,None]
        Pi = np.linalg.inv(Rho)
        
    else:
        if 'beta' in params_init:
            beta = [b.copy() for b in params['beta_init']]
            weight = [np.linalg.pinv(b) for b in beta]
        elif len(data)==2:
            # initialize beta by CCA
            S_xt = np.tensordot(
                np.concatenate([dat-m for dat, m in zip(data,mu)], 1),
                np.concatenate([dat-m for dat, m in zip(data,mu)], 1),
                axes=((0,2),(0,2)))/num_trial/num_time

            S_1 = S_xt[:dims[0],:dims[0]]
            S_12 = S_xt[:dims[0],dims[0]:]
            S_2 = S_xt[dims[0]:,dims[0]:]

            U_1 = linalg.inv(linalg.sqrtm(S_1))
            U_2 = linalg.inv(linalg.sqrtm(S_2))
            u, s, vh = np.linalg.svd(U_1 @ S_12 @ U_2)

            weight = [u[:,:num_f].T @ U_1, vh[:num_f] @ U_2]
            beta = [linalg.inv(U_1) @ u[:,:num_f], linalg.inv(U_2) @ vh[:num_f].T]
        else:
            print("Default initialization only supports 2 populations now.")
            raise
            
        weight = [w*np.sqrt(np.sum(b**2, 0))[:,None] for w, b in zip(weight,beta)]
        beta = [b/np.sqrt(np.sum(b**2, 0)) for b in beta]        

        # initialization on other parameters
        m_z_x = np.concatenate([np.matmul(w, dat-m)[...,None,:]
                 for m, w, dat in zip(mu, weight, data)], -2)
        V_z_x = np.zeros((num_f,len(dims),num_time)*2)

        m_zk_x = m_z_x.transpose((2,0,1,3))
        V_zk_x = np.diagonal(V_z_x,0,1,4).transpose(4,0,1,2,3)

    #     mu = [np.mean(dat - b @ m, 0) 
    #           for dat, m, b in zip(data, m_zk_x, beta)]

        m_eps = [dat - b @ m1 - m2
                 for dat, m1, b, m2 in zip(data, m_zk_x, beta, mu)]
        v_eps = [(np.sum(np.square(m))*(1+lambda_aug)/num_trial 
                  + np.trace(V.reshape(num_f*num_time,num_f*num_time)))
                  # /(d-1)
                 for m, V, d in zip(m_eps, V_zk_x, dims)]

        V_eps_S = [np.tensordot(m,m,axes=((0,2),(0,2)))*(1+lambda_aug)/num_trial/v 
                   + b@np.sum(np.diagonal(V,0,1,3),-1)@b.T/v
                   for m, V, v, b, d in zip(m_eps, V_zk_x, v_eps, beta, dims)]
        V_eps_T = [np.tensordot(m,np.linalg.pinv(V2)@m, axes=((0,1),(0,1)))
                   *(lambda_aug*np.eye(num_time)+1)/d/num_trial 
                   + np.tensordot(V1,b.T@np.linalg.pinv(V2)@b,axes=([0,2],[0,1]))/d
                   for m, V1, V2, b, d in zip(m_eps, V_zk_x, V_eps_S, beta, dims)]
        sd_eps_T = [np.sqrt(np.diag(V)) for V in V_eps_T]
        R_eps_T = [V/sd/sd[:,None] for V, sd in zip(V_eps_T, sd_eps_T)]

        Phi_T = [R*sd*sd[:,None] for sd, R in zip(sd_eps_T, R_eps_T)]
        Gamma_T = [np.linalg.inv(P) for P in Phi_T]

        V_zf = (np.diagonal(V_z_x,0,0,3).transpose((4,0,1,2,3))
             + (m_z_x.reshape((-1,num_f,len(dims)*num_time)).transpose((1,2,0))
             @ m_z_x.reshape((-1,num_f,len(dims)*num_time)).transpose((1,0,2))
             * (lambda_aug*np.eye(2*num_time)+1)/ num_trial)\
             .reshape((num_f,len(dims),num_time,len(dims),num_time)))

        Sigma, Phi_S = _switch_back(
            V_zf.reshape(num_f,len(dims)*num_time,len(dims)*num_time),
            V_eps_S, Gamma_T, Phi_T, beta)
        if make_PD:
            Phi_S = [_make_PD(P) for P in Phi_S]
            for f in np.arange(num_f):
                Sigma[f] = _make_PD(Sigma[f])
        Gamma_S = [np.linalg.inv(P) for P in Phi_S]

        sig = np.sqrt(np.diagonal(Sigma,0,1,2))
        Rho = Sigma/sig[:,None,:]/sig[:,:,None]
        Pi = np.linalg.inv(Rho)
        Omega = Pi/sig[:,None,:]/sig[:,:,None]
    
    cost = (- log_like(data, {'Omega': Omega, 'beta': beta, 'mu': mu, 
                  'Gamma_S': Gamma_S, 'Gamma_T': Gamma_T}, lambda_aug) / num_trial
            + np.sum(np.where(lambda_glasso*np.abs(Pi)>=0, 
                              lambda_glasso*np.abs(Pi), np.inf)))
    
    # EM algorithm
    try: 
        for iter_ldfa in np.arange(max_ldfa):
            Rho_ldfa = Rho.copy() # np.linalg.inv(Pi)
            beta_ldfa = [b.copy() for b in beta]
            cost_ldfa = cost
            start_ldfa = time.time()

            # E-step for Z
            W_z_x = np.zeros((num_f,len(dims),num_time)*2)
            for i, (G_S, G_T, b) in enumerate(zip(Gamma_S, Gamma_T, beta)):
                W_z_x[:,i,:,:,i,:] += G_T[None,:,None,:]*(b.T@G_S@b)[:,None,:,None]
            for j, W in enumerate(Omega):
                W_z_x[j,:,:,j,:,:] += W.reshape(len(dims),num_time,len(dims),num_time)

            V_z_x = np.linalg.inv(
                W_z_x.reshape((num_f*len(dims)*num_time,num_f*len(dims)*num_time))) \
                .reshape((num_f,len(dims),num_time)*2)
            V_zk_x = np.diagonal(V_z_x,0,1,4).transpose(4,0,1,2,3)

            y = np.stack([(b.T @ G_S) @ (dat - m) for dat, m, b, G_S
                           in zip(data, mu, beta, Gamma_S)], axis=-2)
            S_y = np.tensordot(y, y, (0,0)) / num_trial \
                  * (lambda_aug*np.eye(len(dims)*num_time)
                     .reshape(len(dims),num_time,1,len(dims),num_time)+1)
            V_z_y = np.stack([np.tensordot(V_z_x[...,i,:], G_T, [-1, 0])
                              for i, G_T in enumerate(Gamma_T)], -2)
            S_mz = np.tensordot(np.tensordot(V_z_y, S_y, [(-3,-2,-1),(0,1,2)]),
                                V_z_y, [(-3,-2,-1),(-3,-2,-1)])

            for iter_pb in np.arange(10):
                beta_pb = [b.copy() for b in beta]

                # coordinate descent for beta
                S_mz_x_S = [np.tensordot(np.tensordot(y, np.stack(
                    [(np.tensordot(Gamma_T[i], V_z_y[:,i,:,:,j,:], [-1,1]) 
                     * (lambda_aug*(i==j)*np.eye(num_time)+1)[:,None,None,:])
                     for j in np.arange(len(dims))], -2), [(-3,-2,-1),(-3,-2,-1)]),
                    data[i] - mu[i], [(0,1),(0,-1)]) / num_trial
                    for i in np.arange(len(dims))]
                S_mz_S = [np.tensordot(S_mz[:,i,:,:,i,:], G_T, 
                                        [(-3,-1),(0,1)]) 
                          for i, G_T in enumerate(Gamma_T)]

                beta = [S1.T @ np.linalg.inv(S2 + np.tensordot(G_T, V, axes=((0,1),(1,3))))
                        for S1, S2, V, G_T in zip(S_mz_x_S, S_mz_S, V_zk_x, Gamma_T)]

                # fitting Matrix-variate for Phi_S
                V_eps_S = [
                    (np.tensordot((dat-m) @ (G_T*(lambda_aug*np.eye(num_time)+1)), dat-m,
                                   axes=((0,2),(0,2)))/num_trial
                    - b @ S1 - S1.T @ b.T + b @ S2 @ b.T
                    + b @ np.tensordot(V, G_T, [(-3,-1),(0,1)]) @ b.T)/num_time
                    for dat, m, b, G_T, V, S1, S2
                    in zip(data, mu, beta, Gamma_T, V_zk_x, S_mz_x_S, S_mz_S)]
                Phi_S = [V.copy() for V in V_eps_S]
                Gamma_S = [np.linalg.inv(P) for P in Phi_S]

                # fitting Matrix_variate for Phi_T
                y1 = np.stack([(b.T @ G_S) @ (dat - m) for dat, m, b, G_S
                               in zip(data, mu, beta, Gamma_S)], axis=-2)
                S_y_y1 = np.tensordot(y, y1, (0,0)) / num_trial \
                      * (lambda_aug*np.eye(len(dims)*num_time)
                         .reshape(len(dims),num_time,1,len(dims),num_time)+1)

                S_bmz_x_T = [np.tensordot(V_z_y[:,i,:,:,:,:], S_y_y1[:,:,:,:,i,:],
                                          [(0,2,3,4),(3,0,1,2)])
                             for i in np.arange(len(dims))]
                S_bmz_T = [np.tensordot(S_mz[:,i,:,:,i,:], b.T @ G_S @ b, [(0,2),(0,1)])
                           for i, (b, G_S) in enumerate(zip(beta, Gamma_S))]

                V_eps_T = [
                    (np.tensordot(dat-m, G_S@(dat-m), axes=((0,1),(0,1)))/num_trial
                    * (lambda_aug * np.eye(num_time) + 1) - S1 - S1.T + S2
                    + np.tensordot(V, b.T@G_S@b, axes=([0,2],[0,1])))/d
                    for dat, d, m, b, G_S, V, S1, S2 
                    in zip(data, dims, mu, beta, Gamma_S, V_zk_x, S_bmz_x_T, S_bmz_T)]
                sd_eps_T = [np.sqrt(np.diag(V)) for V in V_eps_T]
                R_eps_T = [V/sd/sd[:,None] for V, sd in zip(V_eps_T, sd_eps_T)]
                
                Phi_T = []
                for i, (Rt, sd, d) \
                in enumerate(zip(R_eps_T, sd_eps_T, dims)):
                    Tt, Pt = _temporal_est(Rt, offset_auto)
                    Gamma_T[i] = Tt / sd / sd[:,None]
                    Phi_T.append(Pt * sd * sd[:,None])

                # M-step for mu
                # mu = [np.mean(dat - b @ m, 0) for dat, m, b in zip(data, m_zk_x, beta)]

                beta_diff = [1-np.sum(b1*b2,0)/np.sqrt(np.sum(b1**2,0)*np.sum(b2**2,0)) 
                             for b1, b2 in zip(beta, beta_pb)]
                if np.max(beta_diff) < 1e-12:
                    break 

            # Normalize beta
            V_zf = (np.diagonal(V_z_x,0,0,3)+np.diagonal(S_mz,0,0,3)).transpose(4,0,1,2,3)    
            B = np.concatenate([np.sqrt(np.sum(b**2,0))[:,None] for b in beta], -1)
            Sigma = (V_zf * B[:,:,None,None,None] * B[:,None,None,:,None]) \
                    .reshape((num_f,len(dims)*num_time,len(dims)*num_time))
            beta = [b / np.sqrt(np.sum(b**2, 0)) for b in beta]    

            # Switch forward
        #     Sigma, Phi_S = switch_forward(Sigma, Phi_S, Phi_T, beta)

            # M-step for Pi
            sig = np.sqrt(np.diagonal(Sigma,0,1,2))
            Rho = Sigma/sig[:,None,:]/sig[:,:,None]
            for i, R in enumerate(Rho):
                P = Pi[i].copy()
                core.glasso(P, np.linalg.inv(P), R, lambda_glasso,
                            ths_glasso, max_glasso, ths_lasso, max_lasso)
                Pi[i] = P
            Omega = Pi/sig[:,None,:]/sig[:,:,None]

            # Switch back
        #     Sigma, Phi_S = switch_back(Sigma, Phi_S, Theta_T, beta)
        #     Omega = np.linalg.inv(Sigma)
        #     Theta_S = [np.linalg.inv(P) for P in Phi_S]

            # calculate cost
            cost = (- log_like(data, {'Omega': Omega, 'beta': beta, 'mu': mu, 
                        'Gamma_S': Gamma_S, 'Gamma_T': Gamma_T}, lambda_aug) / num_trial
                    + np.sum(np.where(lambda_glasso*np.abs(Pi)>=0, 
                                      lambda_glasso*np.abs(Pi), np.inf)))

            change_Sigma = np.max(np.abs(Rho - Rho_ldfa))
            change_beta = np.max([1-np.sum(b1*b2,0)/
                                  np.sqrt(np.sum(b1**2,0)*np.sum(b2**2,0)) 
                                  for b1, b2 in zip(beta, beta_ldfa)])
            change_cost = cost_ldfa - cost
            lapse = time.time() - start_ldfa
            if verbose:
                print("%d-th ldfa iter, dcost: %.2e, dRho: %.2e, dbeta: %.2e,"
                      "lapse: %.2fsec."
                      %(iter_ldfa+1, change_cost, change_Sigma, change_beta, lapse))

            if(change_cost < ths_ldfa):
                break
                
    except KeyboardInterrupt:
        return Pi, Rho, {'Omega': Omega, 'beta': beta, 'mu': mu, 
                         'Gamma_S': Gamma_S, 'Gamma_T': Gamma_T}  
        raise

    return Pi, Rho, {'Omega': Omega, 'beta': beta, 'mu': mu, 
                     'Gamma_S': Gamma_S, 'Gamma_T': Gamma_T}  

def log_like(data, params, lambda_aug):
    """Compute the marginal log-likelihood based on estimated parameters.
    
    Parameters
    ----------
    data: list of (N, p_k, T) ndarrays
        Observed data from K areas. Data from each area k consists of p_k-variate
        time-series over T time bins in N trials. 
    params: dict
        The dictionary of the estimated parameters. It provides with the estimation of
            Omega: (num_f, K*T, K*T) ndarray; 
            Gamma_S: a list of (p_k, p_k) ndarrays for k = 1, ..., K;
            Gamma_T: a list of (T, T) ndarrays for k = 1, ..., K;
            beta: a list of (p_k, num_f) ndarrays for k = 1, ..., K; and
            mu: a list of (p_k, T) ndarrays for k = 1, ..., K. 
            
    Returns
    -------
    float
        The sample mean log-likelihood of the given parameters wrt. the data.
            
    """
    Omega = params['Omega']; mu = params['mu']; beta = params['beta']
    Gamma_S = params['Gamma_S']; Gamma_T = params['Gamma_T']
    
    dims = [dat.shape[1] for dat in data]
    num_time = data[0].shape[2]
    num_trial = data[0].shape[0]
    num_f = Omega.shape[0]
    
    W_z_x = np.zeros((num_f,len(dims),num_time)*2)
    for i, (G_S, G_T, b) in enumerate(zip(Gamma_S, Gamma_T, beta)):
        W_z_x[:,i,:,:,i,:] += G_T[None,:,None,:]*(b.T@G_S@b)[:,None,:,None]
    for j, W in enumerate(Omega):
        W_z_x[j,:,:,j,:,:] += W.reshape(len(dims),num_time,len(dims),num_time)

    V_z_x = np.linalg.inv(
        W_z_x.reshape((num_f*len(dims)*num_time,num_f*len(dims)*num_time))) \
        .reshape((num_f,len(dims),num_time)*2)
    
    y = np.stack([(b.T @ G_S) @ (dat - m) for dat, m, b, G_S
               in zip(data, mu, beta, Gamma_S)], axis=-2)
    S_y = np.tensordot(y, y, (0,0)) / num_trial \
          * (lambda_aug*np.eye(len(dims)*num_time)
             .reshape(len(dims),num_time,1,len(dims),num_time)+1)
    V_z_y = np.matmul(V_z_x, np.array(Gamma_T)[:,:,:,...],
                      axes=[(-3,-1),(-2,-1),(-3,-1)])
    
    S_mz = np.tensordot(np.tensordot(V_z_y, S_y, [(-3,-2,-1),(0,1,2)]),
                        V_z_y, [(-3,-2,-1),(-3,-2,-1)])
    S_bmz_x_T = np.sum(np.diagonal(
    np.diagonal(np.tensordot(V_z_y, S_y, [(-3,-2,-1),(0,1,2)]),0,1,4),
    0,0,2),-1).transpose(2,0,1)
    S_bmz_T = [np.tensordot(S_mz[:,i,:,:,i,:], b.T @ G_S @ b, [(0,2),(0,1)])
        for i, (b, G_S) in enumerate(zip(beta, Gamma_S))]
    
    m_z_x = np.tensordot(y, V_z_y, [(-3,-2,-1),(-3,-2,-1)])
    m_zk_x = m_z_x.transpose((2,0,1,3))

    S_zf = (np.diagonal(S_mz,0,0,3)).transpose(4,0,1,2,3)
    S_eps_T = [
        ( np.tensordot(dat-m,G_S@(dat-m),axes=((0,1),(0,1)))/num_trial
        * (lambda_aug*np.eye(num_time)+1)
        - S1 - S1.T + S2)
        for i, (dat, d, m, b, G_S, S1, S2) 
        in enumerate(zip(data, dims, mu, beta, Gamma_S, S_bmz_x_T, S_bmz_T))
    ]
    
    return num_trial * (
    + np.sum(np.linalg.slogdet(Omega)[1])
    + np.sum(
        [d * np.linalg.slogdet(G_T)[1] 
         + num_time * np.linalg.slogdet(G_S)[1]
        for d, G_T, G_S in zip(dims, Gamma_T, Gamma_S)])
    + np.linalg.slogdet(V_z_x.reshape((num_f*len(dims)*num_time,)*2))[1]
    - np.sum(S_zf.reshape(num_f,len(dims)*num_time,len(dims)*num_time)*Omega)
    - np.sum([np.sum(S_T*G_T) for S_T, G_T in zip(S_eps_T, Gamma_T)]))

def _dof(params):
    """Estimate the effective degree of freedom in the parameters"""
    return (np.sum(params['Omega'] != 0) 
            + np.sum([np.sum(G_T != 0) for G_T in params['Gamma_T']]))

def AIC(data, params, lambda_aug=0):
    """Calculate Akaike Information Criterion"""
    return - 2*log_like(data, params, lambda_aug) + 2*_dof(params)

def BIC(data, params, lambda_aug=0):
    """Calculate Bayesian Information Criterion"""
    num_trial = data[0].shape[0]
    return - 2*log_like(data, params, lambda_aug) + np.log(num_trial)*_dof(params)
    
def cross_validate(data, num_f, lambdas_cross, offset_cross, **kwargs):
    """Cross validation to determine cross lambda"""
    nl = len(lambdas_cross)
    loglikes = np.zeros(nl)
    num_trial = data[0].shape[0]
    num_train= np.int(num_trial*0.8)
    
    sample_id = np.random.choice(num_trial, num_trial)
    dtrain = [dat[sample_id[:num_train],:,:] for dat in data] 
    dval = [dat[sample_id[:num_train],:,:] for dat in data] 
    
    for il in range(nl):
        lambda_cross = lambdas_cross[il]
        print('current lambda', lambda_cross)
        Pi, Rho, params = fit(dtrain, num_f, lambda_cross, offset_cross, **kwargs)
        ll = log_like(dval, params)
        loglikes[il] = ll
    return loglikes 

def imshow(image, vmin=None, vmax=None, cmap='RdBu', time=None, identity=False, **kwargs):
    """Color plot function using ``matplotlib.pyplot.imshow`` with dedicated features
    for illustrating dynamic connectivity between two multivariate time-series.
    
    Parameters
    ----------
    image: (M, N) ndarray
        An image data with scalar data, which are colormapped.
    vmin, vmax: float, optional
        The range of the scalar values in image data which the colormap covers. If vmax
        is not given, it is automatically set by the maximum absolute value in image data
        by default. If vmin is not given, it is automatically set by -vmax.
    cmap: string or Colormap, optional
        The Colormap instance or registered colormap name. The default value is 'RdBu'.
    time: list or tuple of two floats, optional
        The time range of a dynamic association the function plots. 
    identity: boolean, optional
        Switch for drawing the line of simultaneous association in a dynamic association
        plot. The parameter time should be given alongside.
    **kwargs
        Keyword parameters for the function ``matplotlib.pyplot.imshow``. 
        
    See Also
    --------
    matplotlib.pyplot.imshow
    
    Examples
    --------
    imshow(Pi[0,0:T,T:2*T], time=(0,T), identity=True)
    """
    if time:
        assert(image.shape[0] == image.shape[1])
        kwargs['extent'] = [time[0], time[1], time[1], time[0]]
    
    image = np.array(image).astype(float)
    assert(image.ndim == 2)

    # get vmin, vmax
    if vmax is None:
        vmax = np.maximum(np.max(np.abs(image)), 1e-6)
    if vmin is None:
        vmin = -vmax
    
    # get figure    
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs) 
        
    if identity and time:
        plt.plot([time[0], time[1]], [time[0], time[1]], linewidth = 0.3, color='black')
        

# def fit_Pi(data, params, lambda_cross, offset_cross, 
#            lambda_auto=None, offset_auto=None, 
#            ths_ldfa=1e-2, max_ldfa=1000, ths_glasso=1e-8, max_glasso=1000, 
#            ths_lasso=1e-8, max_lasso=1000, verbose=False):
#     """Conditional estimate of Pi given other parameters
    
#     Parameters
#     ----------
#     data: list of (N, p_k, T) ndarrays
#         Observed data from K areas. Data from each area k consists of p_k-variate
#         time-series over T time bins in N trials. 
#     params: dict
#         The dictionary of the given other parameters, which are
#             Gamma_S: a list of (p_k, p_k) ndarrays for k = 1, ..., K;
#             Gamma_T: a list of (T, T) ndarrays for k = 1, ..., K;
#             beta: a list of (p_k, num_f) ndarrays for k = 1, ..., K; and
#             mu: a list of (p_k, T) ndarrays for k = 1, ..., K. 
#     lambda_cross, lambda_auto: float
#         The sparsity penalty parameter for the inverse cross-correlation and inverse
#         auto-correlation matrix, respectively. The default value for lambda_auto is 0.
#     offset_cross, offset_auto: int
#         The bandwidth parameter for the inverse cross-correlation matrix and inverse
#         auto-correlation matrix, respectively. The default value for offset_auto is the
#         given value of offset_cross.
#     ths_ldfa, ths_glasso, ths_lasso: float, optional
#         The threshold values for deciding the convergence of the main iteration, the
#         glasso iteration, and the lasso iteration, respectively.
#     max_ldfa, max_glasso, max_lasso: int, optional
#         The maximum number of iteration for the main iteration, the glasso iteration,
#         and the lasso iteration, respectively.
#     beta_init: list of (p_k, num_f) ndarrays, optional
#         Custom initial values for beta. If not given, beta is initialized by CCA.
#     make_PD: boolean, optional
#         Switch for manual positive definitization. If data does not generate a positive
#         definite estimate of the covariance matrix, ``make_PD = True`` helps with 
#         maintaining the matrix positive definite throughout the fitting algorithm. The
#         default value is False for the sake of running time.
#     verbose: boolean, optional
#         Swith for vocal feedback throughout the fitting algorithm. The default value is
#         False.
   
#     Returns
#     -------
#     Pi: (K*T, K*T) ndarray
#         The estimated sparse inverse correlation matrix.
#     Rho: (K*T, K*T) ndarray
#         The estimated correlation matrix before sparsification. Note that Rho != Pi^{-1}.
    
#     Examples
#     --------
#     Pi, Rho =\
#         fit(data, params, lambda_cross, offset_cross, lambda_auto, offset_auto)
                 
#     .. _[1] A. Anonymous. (2020). Latent Dynamic Factor Analysis of High-Dimensional Neural Recordings. Submitted to NeurIPS2020.
    
#     """
    
#     Omega = params['Omega'].copy()
#     beta = params['beta']; Gamma_S = params['Gamma_S']; Gamma_T = params['Gamma_T']
#     mu= [np.mean(dat, 0) for dat in data]
    
#     dims = [data[0].shape[1], data[1].shape[1]]
#     num_time = data[0].shape[2]
#     num_trial = data[0].shape[0]
#     num_f = Omega.shape[0]
    
#     Sigma = np.linalg.inv(Omega)
#     sig = np.sqrt(np.diagonal(Sigma,0,1,2))
#     Rho = Sigma/sig[:,None,:]/sig[:,:,None]
#     Pi = Omega*sig[:,None,:]*sig[:,:,None]

#     # get full_graph
#     if lambda_auto is None:
#         lambda_auto = lambda_cross
#     if offset_auto is None:
#         offset_auto = offset_cross
#     lambda_glasso_auto = _generate_lambda_glasso(num_time, lambda_auto, 
#                                                  offset_auto)
#     lambda_glasso_cross = _generate_lambda_glasso(num_time, lambda_cross,
#                                                   offset_cross)
#     lambda_glasso = np.array(np.block(
#         [[lambda_glasso_auto if j==i else lambda_glasso_cross
#           for j, _ in enumerate(data)]
#          for i, _ in enumerate(data)])) 
    
#     cost = (- log_like(data, {'Omega': Omega, 'beta': beta, 'mu': mu, 
#                               'Gamma_S': Gamma_S, 'Gamma_T': Gamma_T}) / num_trial + 
#               np.sum(np.where(lambda_glasso*np.abs(Pi)>=0, 
#                               lambda_glasso*np.abs(Pi), np.inf)))
    
#     for iter_ldfa in np.arange(max_ldfa):
#         iPi_ldfa = np.linalg.inv(Pi)
#         beta_ldfa = [b.copy() for b in beta]
#         cost_ldfa = cost
#         start_ldfa = time.time()
        
#         # E-step
#         W_z_x = np.zeros((num_f,len(dims),num_time)*2)
#         for i, (G_S, G_T, b) in enumerate(zip(Gamma_S, Gamma_T, beta)):
#             W_z_x[:,i,:,:,i,:] += G_T[None,:,None,:]*(b.T@G_S@b)[:,None,:,None]
#         for j, W in enumerate(Omega):
#             W_z_x[j,:,:,j,:,:] += W.reshape(len(dims),num_time,len(dims),num_time)
#         V_z_x = np.linalg.inv(
#             W_z_x.reshape((num_f*len(dims)*num_time,num_f*len(dims)*num_time))) \
#             .reshape((num_f,len(dims),num_time)*2)
#         m_z_x = np.tensordot(
#             np.concatenate([(b.T @ G_S @ (dat - m) @ G_T)[...,None,:]
#                 for dat, G_T, G_S, b, m in zip(data, Gamma_T, Gamma_S, beta, mu)], -2),
#             V_z_x, axes=((-3,-2,-1), (0,1,2)))

#         V_zf = (np.diagonal(V_z_x,0,0,3).transpose((4,0,1,2,3))
#              + (m_z_x.reshape((-1,num_f,len(dims)*num_time)).transpose((1,2,0))
#              @ m_z_x.reshape((-1,num_f,len(dims)*num_time)).transpose((1,0,2))
#              / num_trial).reshape((num_f,len(dims),num_time,len(dims),num_time)))

#         # M-step for Pi
#         Sigma = V_zf.reshape((num_f,len(dims)*num_time,len(dims)*num_time))
#         sig = np.sqrt(np.diagonal(Sigma,0,1,2))
#         Rho = Sigma/sig[:,None,:]/sig[:,:,None]
#         for i, (P, R) in enumerate(zip(Pi, Rho)):
#             core.glasso(P, np.linalg.inv(P), R, lambda_glasso,
#                         ths_glasso, max_glasso, ths_lasso, max_lasso)
#             Pi[i] = P
#         Omega = Pi/sig[:,None,:]/sig[:,:,None]
        
#         # calculate cost
#         cost = (- log_like(data,{'Omega': Omega, 'beta': beta, 'mu': mu, 
#                                  'Gamma_S': Gamma_S, 'Gamma_T': Gamma_T}) / num_trial 
#                 + np.sum(np.where(lambda_glasso*np.abs(Pi)>=0, 
#                                   lambda_glasso*np.abs(Pi), np.inf)))

#         change_Sigma = np.max(np.abs(np.linalg.inv(Pi) - iPi_ldfa))
#         change_beta = np.max([1-np.sum(b1*b2,0)/np.sqrt(np.sum(b1**2,0)*np.sum(b2**2,0)) 
#                          for b1, b2 in zip(beta, beta_ldfa)])
#         change_cost = cost_ldfa - cost
#         lapse = time.time() - start_ldfa
#         if verbose:
#             print("%d-th ldfa iter, dcost: %.2e, dRho: %.2e, dbeta: %.2e,"
#                   "lapse: %.2fsec."
#                   %(iter_ldfa+1, change_cost, change_Sigma, change_beta, lapse))

#         if(change_Sigma < ths_ldfa):
#             break
    
#     return Pi, Rho
