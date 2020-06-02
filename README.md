# LDFA-H: Latent Dynamic Factor Analysis of High-dimensional time series

This repo consists of 
1. Python package `ldfa` which implements LDFA-H [[1](#20)]
2. Experimental data analyzed in [[1](#ANON20)]
3. Reproducible IPython notebooks for simulation and experimental data analysis in [[1](#ANON20)]

## Install

### Prerequisite

Package `ldfa` requires:
1. `Python` >= 3.5 
2. `numpy` >= 1.8
3. `matplotlib` and `scipy`.

### `Git` clone

Clone this repo through github:
```bash
git clone https://github.com/AutoAnonymous/ldfa.git
```

### `Python` install

Install package `ldfa` using setup.py script:
```bash
python setup.py install
```

## Experimental data 

The data are available in `/data/`. The data consists of `pfc_lfp_bred.mat` and `v4_lfp_bred.mat` which are the beta band-passed filtered LFP in PFC and V4, respectively.

## Reproducible Ipython notebooks

The scripts are available in `/example/`. The script for simulation analysis is provided in `Python` notebook `3.1 CDFA versus existing methods in addressing noise auto-correlation.ipynb` with an accessory `MATLAB` notebook `3.1 DKCCA on simulated data.ipynb`. To run `DKCCA`, a separate installation of `MATLAB` package `DKCCA` is required [[2](#RKBMK18)].

## References

<a name="ANON20"> [1] Bong, H., Liu, Z., Ren, Z., Smith, M., Ventura, V., & Kass, R. E. (2020). Latent Dynamic Factor Analysis of High-Dimensional Neural Recordings. *Submitted to NeurIPS2020*. </a>

<a name="RKBMK18"> [2] Rodu, J., Klein, N., Brincat, S. L., Miller, E. K., & Kass, R. E. (2018). Detecting multivariate cross-correlation between brain regions. *Journal of neurophysiology*, 120(4), 1962-1972. </a>
