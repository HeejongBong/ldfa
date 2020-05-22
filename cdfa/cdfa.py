import time

import numpy as np
from scipy import linalg

import miccs.optimize as core

def _generate_lambda_glasso(bin_num, lambda_glasso, offset, lambda_diag=None):
    lambda_glasso_out = np.full((bin_num, bin_num), -1) + (1+lambda_glasso) * \
           (np.abs(np.arange(bin_num) - np.arange(bin_num)[:,np.newaxis]) <= offset)
    if lambda_diag:
        lambda_glasso_out[np.arange(bin_num), np.arange(bin_num)] = lambda_diag
    return lambda_glasso_out

def fit(data, lambda_cross, offset_cross, 
        lambda_auto=None, offset_auto=None, 
        ths_dfa=1e-3, max_dfa=1000, ths_glasso=1e-5, max_glasso=1000,
        ths_lasso=1e-5, max_lasso=1000, beta_init=None,
        switch=False, verbose=False):
    
    num_time = data[0].shape[0]
    dims = [dat.shape[1] for dat in data]
    num_trial = data[0].shape[2]

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
    
    # initialization by CCA
    m_x = [np.mean(dat, -1) for dat in data]
    S_xt = np.cov(*[dat.transpose([1,0,2]).reshape([d,-1])
                    for dat, d in zip(data, dims)])
    
    if beta_init:
        beta = [b.copy() for b in beta_init]
    else:
        S_1 = S_xt[:dims[0],:dims[0]]
        S_12 = S_xt[:dims[0],dims[0]:]
        S_2 = S_xt[dims[0]:,dims[0]:]
        U_1= linalg.inv(linalg.sqrtm(S_1))
        U_2 = linalg.inv(linalg.sqrtm(S_2))
        u, s, vh = np.linalg.svd(U_1 @ S_12 @ U_2)
        beta = [U_1 @ u[:,0], U_2 @ vh[0]]
    beta = [b/np.sqrt(np.sum(b**2)) for b in beta]

    m_z_x = np.concatenate([np.matmul(b, dat-m[...,None]) 
             for m, b, dat in zip(m_x, beta, data)])
    V_z_x = np.zeros((num_time*len(dims), num_time*len(dims)))
    m_zk_x = np.reshape(m_z_x, (len(dims), num_time, -1))
    V_zk_x = np.diagonal(V_z_x.reshape((len(dims),num_time,len(dims),num_time)),
                         0,0,2).transpose((2,0,1))

    mu = [np.mean(dat - m[:,None,:] * b[:,None], -1) 
          for dat, m, b in zip(data, m_zk_x, beta)]

    m_eps = [dat - m[:,None,:] * b[:,None] 
             - np.mean(dat - m[:,None,:] * b[:,None], -1)[...,None]
             for dat, m, b in zip(data, m_zk_x, beta)]
    v_eps = [(np.sum(np.square(m))/num_trial + np.sum(np.diag(V))) #/(d-1)
             for m, V, d in zip(m_eps, V_zk_x, dims)]

    V_eps_S = [np.tensordot(m.transpose([1,0,2]).reshape([d,-1]),
                            m.transpose([1,0,2]).reshape([d,-1]).T, 1)
               /num_trial/v + np.sum(np.diag(V))*b*b[:,None]/v
               for m, V, v, b, d in zip(m_eps, V_zk_x, v_eps, beta, dims)]
    V_eps_T = [np.tensordot(m.reshape([num_time,-1]),
                            (np.linalg.pinv(V2) @ m).reshape([num_time,-1]).T, 1)
               /d/num_trial + (b @ np.linalg.pinv(V2) @ b) * V1 / d
               for m, V1, V2, b, d in zip(m_eps, V_zk_x, V_eps_S, beta, dims)]
    sd_eps_T = [np.sqrt(np.diag(V)) for V in V_eps_T]
    R_eps_T = [V/sd/sd[:,None] for V, sd in zip(V_eps_T, sd_eps_T)]
    Phi_T = [R*sd*sd[:,None] for sd, R in zip(sd_eps_T, R_eps_T)]
    Theta_T = [np.linalg.inv(P) for P in Phi_T]

    V_z = np.concatenate(m_zk_x,0) @ np.concatenate(m_zk_x).T / num_trial + V_z_x
    Sigma, Phi_S = switch_forward(V_z, V_eps_S, Phi_T, beta)

    sig = np.sqrt(np.diag(Sigma))
    Rho = Sigma/sig/sig[:,None] 
    Pi = np.linalg.inv(Rho)
    Omega = Pi/sig/sig[:,None]
    Sigma = np.linalg.inv(Omega)

    Sigma, Phi_S = switch_back(Sigma, Phi_S, Theta_T, Phi_T, beta)
    Omega = np.linalg.inv(Sigma)
    Theta_S = [np.linalg.inv(P) for P in Phi_S]
    
    # EM algorithm
    for iter_dfa in np.arange(max_dfa):
        invPi_last = np.linalg.inv(Pi)
        beta_ldfa = [b.copy() for b in beta]
        start_dfa = time.time()

        # E-step for Z
        V_z_x = linalg.inv(Omega + linalg.block_diag(*[b @ Ts @ b * Tt 
            for b, Ts, Tt in zip(beta, Theta_S, Theta_T)]))
        m_z_x = V_z_x @ np.concatenate([Tt @ np.tensordot(Ts@b, dat-m[...,None], (0,1))
            for Tt,Ts,b,m,dat in zip(Theta_T,Theta_S,beta,mu,data)])
        m_zk_x = np.reshape(m_z_x, (len(dims), num_time, -1))
        V_zk_x = np.diagonal(V_z_x.reshape(
            (len(dims),num_time,len(dims),num_time)),0,0,2).transpose((2,0,1))

        for iter_pb in np.arange(10):
            beta_last = [b.copy() for b in beta]

            # E-step for eps
            m_eps = [dat - m[:,None,:] * b[:,None] 
                     - np.mean(dat - m[:,None,:] * b[:,None], -1)[...,None]
                     for dat, m, b in zip(data, m_zk_x, beta)]
            v_eps = [(np.sum(np.square(m))/num_trial + np.sum(np.diag(V))) #/(d-1)
                     for m, V, d in zip(m_eps, V_zk_x, dims)]

            # fitting Matrix-variate for Phi_S
            V_eps_S = [np.tensordot(m.transpose([1,0,2]).reshape([d,-1]),
                                    m.transpose([1,0,2]).reshape([d,-1]).T, 1)
                       /num_trial/v + np.sum(np.diag(V))*b*b[:,None]/v
                       for m, V, v, b, d in zip(m_eps, V_zk_x, v_eps, beta, dims)]
            Phi_S = [V.copy() for V in V_eps_S]

            # fitting Matrix_variate for Phi_T
            V_eps_T = [np.tensordot(m.reshape([num_time,-1]),
                                    (np.linalg.inv(V2) @ m).reshape([num_time,-1]).T, 1)
                       /d/num_trial + (b @ np.linalg.inv(V2) @ b) * V1 / d
                       for m, V1, V2, b, d in zip(m_eps, V_zk_x, V_eps_S, beta, dims)]

            sd_eps_T = [np.sqrt(np.diag(V)) for V in V_eps_T]
            R_eps_T = [V/sd/sd[:,None] for V, sd in zip(V_eps_T, sd_eps_T)]
            for i, (Rt, sd, d) \
            in enumerate(zip(R_eps_T, sd_eps_T, dims)):
                Tt, Pt = temporal_est(Rt, offset_auto)

                Theta_T[i] = Tt / sd / sd[:,None]
                Phi_T[i] = Pt * sd * sd[:,None]

            # coordinate descent for beta
            Cov_mz_X = [(m-np.mean(m, -1)[...,None]) @ 
                        (dat - np.mean(dat, -1)[...,None]).transpose([0, 2, 1])
                        /num_trial for m, dat in zip(m_zk_x, data)]
            Cov_z = [(m-np.mean(m, -1)[...,None]) @ (m-np.mean(m, -1)[...,None]).T
                       /num_trial + V for m, V in zip(m_zk_x, V_zk_x)]

            beta = [np.sum(T[...,None] * C1, (0, 1)) / np.sum(T * C2)
                    for T, C1, C2 in zip(Theta_T, Cov_mz_X, Cov_z)]

            beta_diff = [1-b1@b2/np.sqrt(np.sum(b1**2)*np.sum(b2**2)) 
                         for b1, b2 in zip(beta, beta_last)]
            if np.max(beta_diff) < 1e-12:
                break

        # M-step for mu
        mu = [np.mean(dat - m[:,None,:] * b[:,None], -1) 
              for dat, m, b in zip(data, m_zk_x, beta)]

        # M-step for Sigma
        V_z = np.cov(m_z_x) + V_z_x
        Sigma = V_z.copy()

        # Normalize beta
        B = np.concatenate([np.full(num_time, np.sqrt(np.sum(b**2)))
                            for b in beta])
        Sigma = Sigma * B * B[:,None]
        beta = [b / np.sqrt(np.sum(b**2)) for b in beta]

        # Switch forward
        if switch:
            Sigma, Phi_S = switch_forward(Sigma, Phi_S, Phi_T, beta)

        # M-step for Pi
        sig = np.sqrt(np.diag(Sigma))
        Rho = Sigma/sig/sig[:,None]
        core.glasso(Pi, np.linalg.inv(Pi), Rho, lambda_glasso,
                    ths_glasso, max_glasso, ths_glasso, max_lasso)
        Omega = Pi/sig/sig[:,None]
        Sigma = np.linalg.inv(Omega)

        # Switch back
        if switch:
            Sigma, Phi_S = switch_back(Sigma, Phi_S, Theta_T, Phi_T, beta)
            Omega = np.linalg.inv(Sigma)
            Theta_S = [np.linalg.inv(P) for P in Phi_S]  
        
        # Check for convergence
        change_Sigma = np.max(np.abs(np.linalg.inv(Pi) - invPi_last))
        change_beta = np.max([1-b1@b2/np.sqrt(np.sum(b1**2)*np.sum(b2**2)) 
                       for b1, b2 in zip(beta, beta_ldfa)])
        lapse = time.time() - start_dfa
        if verbose:
            print("%d-th dfa iter, change: %f, change_beta: %.2e, lapse: %.2fsec."
                  %(iter_dfa+1, change_Sigma, change_beta, lapse))
        if change_Sigma < ths_dfa:
            break
            
    return Pi, Rho, {'Omega': Omega, 'beta': beta, 'mu': mu, 
                     'Theta_S': Theta_S, 'Theta_T': Theta_T}   

def fit_Pi(data, params, lambda_cross, offset_cross, 
           lambda_auto=None, offset_auto=None, Pi_init=None,
           ths_glasso=1e-5, max_glasso=1000, ths_lasso=1e-5, max_lasso=1000, 
           switch=False, verbose=False):
    
    Omega = params['Omega']; mu = params['mu']; beta = params['beta']
    Theta_S = params['Theta_S']; Theta_T = params['Theta_T']
    
    num_time = data[0].shape[0]
    
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
    
    # E-step
    V_z_x = linalg.inv(Omega + linalg.block_diag(*[b @ Ts @ b * Tt 
            for b, Ts, Tt in zip(beta, Theta_S, Theta_T)]))
    m_z_x = V_z_x @ np.concatenate([Tt @ np.tensordot(Ts@b, dat-m[...,None], (0,1))
        for Tt,Ts,b,m,dat in zip(Theta_T,Theta_S,beta,mu,data)])
    
    # M-step for Sigma
    V_z = np.cov(m_z_x) + V_z_x
    Sigma = V_z.copy()
    
    # Switch forward
    if switch:
        Sigma, Phi_S = switch_forward(Sigma, Phi_S, Phi_T, beta)
            
    # M-step for Pi
    sig = np.sqrt(np.diag(Sigma))
    Rho = Sigma/sig/sig[:,None]
    if Pi_init:
        Pi = Pi_init.copy()
        invPi = np.linalg.inv(Pi)
    else: 
        invPi = Rho.copy()
        Pi = np.linalg.inv(Rho)
    core.glasso(Pi, invPi, Rho, lambda_glasso,
                ths_glasso, max_glasso, ths_glasso, max_lasso)
    
    # Switch back
#     if switch:
#         Sigma, Phi_S = switch_back(Sigma, Phi_S, Theta_T, Phi_T, beta)
#         Omega = np.linalg.inv(Sigma)
#         Theta_S = [np.linalg.inv(P) for P in Phi_S] 
    
    return Pi, Rho

def temporal_est(V_eps_T, ar_order):
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
#     Theta_T_hat = np.linalg.inv(Psi_T_hat)
    
    Theta_T_hat = (np.eye(num_time)-Amatrix).T @ np.diag(1/resids) \
                  @ (np.eye(num_time)-Amatrix)
    Psi_T_hat = np.linalg.pinv(Theta_T_hat)
    
    return Theta_T_hat, Psi_T_hat

def switch_forward(Sigma, Phi_S, Phi_T, beta):
    U_S = [u for u,_,_ in [np.linalg.svd(b[:,None]) for b in beta]]
    UtPU = [U.T @ P @ U for U, P in zip(U_S, Phi_S)]
    ls = [(X[0,0]-X[0,1:]@np.linalg.inv(X[1:,1:])@X[1:,0])
          /np.sum(b**2)
          for X, b in zip(UtPU, beta)]
    
    return (Sigma + linalg.block_diag(*[l*P for l, P in zip(ls, Phi_T)]), 
            [P - b * b[:,None] * l for b, P, l in zip(beta, Phi_S, ls)])

def switch_back(Sigma, Phi_S, Theta_T, Phi_T, beta):
    l = np.min(np.linalg.eig(linalg.sqrtm(Sigma) 
    @ linalg.block_diag(*Theta_T) @ linalg.sqrtm(Sigma))[0])/2
    
    return (Sigma - l*linalg.block_diag(*Phi_T), 
            [P + l*b*b[:,None] for P, b in zip(Phi_S, beta)])  
    
def imshow(image, vmin=None, vmax=None, cmap='RdBu', time=None, identity=False, **kwargs):
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
        import matplotlib.pyplot as plt
        plt.plot([time[0], time[1]], [time[0], time[1]], linewidth = 0.3, color='black')
