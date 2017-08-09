#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: maxL.py
# Author: Hongyu Li <lhy88562189@gmail.com>
# Date: 07.08.2017
# Last Modified: 07.08.2017
# ============================================================================
#  DESCRIPTION: ---
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
# ORGANIZATION:
#      VERSION: 0.0
# ============================================================================
import numpy as np
import util_angle
import util_sample_A as util_sample
from scipy.optimize import minimize


def likelihood(pars, Psai_obs=None, eps_obs=None,
               theta=None, phi=None,
               size=3000000, bins=50, seed=88562189):
    np.random.seed(seed=seed)
    mu_zeta, sigma_zeta, mu_eta, sigma_eta, mu_Psai_int, sigma_Psai_int = pars
    if theta is None:
        theta, phi = util_angle.get_view_angle(size)
    zeta = util_sample.get_sample(mu_zeta, sigma_zeta,
                                  boundary=[0.5, 1.0], size=size)
    eta = util_sample.get_sample(mu_eta, sigma_eta,
                                 boundary=[0.5, 1.0], size=size)

    ksai = eta * zeta
    Psai_int = util_sample.get_sample(mu_Psai_int, sigma_Psai_int,
                                      boundary=[0.0, np.pi/2.0], size=size)
    Psai, eps = util_angle.get_Psai(theta, phi, zeta, ksai, Psai_int)
    H, xedges, yedges = \
        np.histogram2d(eps, Psai, range=[[0.0, 1.0], [0.0, np.pi/2.0]],
                       normed=True, bins=bins)
    H = H.clip(1e-8, None)
    i_eps = (np.floor(eps_obs / (1.0 / bins))).astype(int)
    i_Psai = (np.floor(Psai_obs / (np.pi/2.0 / bins))).astype(int)
    lnprob = -np.log(H[i_eps, i_Psai]).sum()
    print pars, lnprob
    if np.isnan(lnprob):
        return np.inf
    return lnprob


def run_minimize(Psai_obs, eps_obs, p0, size=3000000, bins=50, bounds=None,
                 options=None, seed=88562189):
    np.random.seed(seed=seed)
    theta, phi = util_angle.get_view_angle(size)
    if bounds is None:
        bounds = [(0.0, 1.0), (0.03, 10.0), (0.0, 1.0), (0.03, 10.0),
                  (0.0, np.pi/2.0), (0.03, 20.0)]
    if options is None:
        options = {'maxiter': 1000, 'disp': True, 'eps': 0.0001}
    res = minimize(likelihood, p0,
                   args=(Psai_obs, eps_obs, theta, phi, size, bins, seed),
                   bounds=bounds, method='L-BFGS-B', options=options)
    return res
