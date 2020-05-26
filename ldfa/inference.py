import numpy as np
from scipy import ndimage

def h(pvals):
    return 2*(pvals > 0.5).astype(float)

def s(pvals):
    return 1-pvals

def g(pvals):
    return np.minimum(pvals, s(pvals))

def sinv(tdpvals):
    return 1-tdpvals

def sdot(pvals):
    return -1

def fdp_hat(pvals, mask):
    if np.sum(mask) == 0:
        return 0
    else:
        return (2 + np.sum(mask*h(pvals))) / (1 + np.sum(mask))
    
def score_fn(pvals, mask, steps_em=5, sigma=1, mux_init=None):
    tdpvals_0 = np.where(mask, g(pvals), pvals)
    tdpvals_1 = sinv(tdpvals_0)
    if mux_init is None:
        mux_init = np.mean(-np.log(tdpvals_0))
    mux = np.full(pvals.shape, mux_init)

    for _ in range(steps_em):
        imputed_logpvals = ((tdpvals_0**(1/mux-1)*(-np.log(tdpvals_0)) +
                             tdpvals_1**(1/mux-1)*(-np.log(tdpvals_1))) /
                            (tdpvals_0**(1/mux-1)+tdpvals_1**(1/mux-1)/(-sdot(tdpvals_1))))

        mux = ndimage.gaussian_filter(imputed_logpvals, sigma)
        
    return mux

def STAR_seq_step(pvals, alphas = [0.05], prop_carve = 0.2, roi = None, **kwargs):
    if isinstance(alphas, float):
        alphas = [alphas] 
    alphas = np.array(alphas)
    
    if roi is None:
        roi = np.full(pvals.shape, True)
        
    boi = np.any(roi > np.stack([
        np.concatenate([np.zeros(roi[:1].shape), roi[:-1]], 0),
        np.concatenate([roi[1:], np.zeros(roi[-1:].shape)], 0),
        np.concatenate([np.zeros(roi[:,:1].shape), roi[:,:-1]], 1),
        np.concatenate([roi[:,1:], np.zeros(roi[:,-1:].shape)], 1)], 0),0)
        
    # output value
    masks = np.full(alphas.shape + pvals.shape, False)
    scores = np.zeros(alphas.shape + pvals.shape)
    fdps = np.zeros(alphas.shape)
    Rs = np.zeros(alphas.shape)
            
    # initial value        
    mask = roi.copy()
    boundary = boi.copy()
    
    fdp = fdp_hat(pvals, mask)
    R = np.sum(mask)
    R_min = R * (1-prop_carve)
    
    score = score_fn(pvals, mask, **kwargs)
    alpha_last = np.inf
    
    while np.any(alphas < alpha_last):  
        alpha = np.max(alphas[alphas < alpha_last])
        
        while fdp > alpha:
            min_ind = np.unravel_index(
                np.argmin(score + np.where(mask & boundary, 0, np.inf)), 
                mask.shape)
            mask[min_ind] = False
            if min_ind[0] > 0:
                boundary[min_ind[0]-1, min_ind[1]] = True
            if min_ind[0] < pvals.shape[0]-1:
                boundary[min_ind[0]+1, min_ind[1]] = True
            if min_ind[1] > 0:
                boundary[min_ind[0], min_ind[1]-1] = True
            if min_ind[1] < pvals.shape[1]-1:
                boundary[min_ind[0], min_ind[1]+1] = True

            fdp = fdp_hat(pvals, mask)
            R = np.sum(mask)
                        
            if 2 / (1 + R) > alpha:
                scores[alphas < alpha_last] = score
                return Rs, fdps, scores, masks

            if R <= R_min:
                score = score_fn(pvals, mask, **kwargs)
                R_min = R * (1-prop_carve)
        
        masks[np.logical_and(alphas >= fdp, alphas < alpha_last)] = mask
        scores[np.logical_and(alphas >= fdp, alphas < alpha_last)] = score
        fdps[np.logical_and(alphas >= fdp, alphas < alpha_last)] = fdp
        Rs[np.logical_and(alphas >= fdp, alphas < alpha_last)] = R
        
        alpha_last = fdp
            
    return Rs, fdps, scores, masks