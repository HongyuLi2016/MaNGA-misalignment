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
import util_sample_B as util_sample
from scipy.optimize import minimize


def likelihood(pars, Psai_obs=None, eps_obs=None,
               size=300000, bins=100):
    mu_zeta, sigma_zeta, mu_eta, sigma_eta, mu_Psai_int, sigma_Psai_int = pars

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
    i_eps = (np.floor(eps_obs / (1.0 / bins))).astype(int)
    i_Psai = (np.floor(Psai_obs / (np.pi/2.0 / bins))).astype(int)
    good = np.zeros_like(H, dtype=bool)
    good[i_eps, i_Psai] = True
    good *= H > 0.0
    lnprob = -np.log(H[good]).sum()
    # print pars, lnprob
    return lnprob


def run_minimize(Psai_obs, eps_obs, p0, size=300000, bins=100, bounds=None,
                 options=None):
    if bounds is None:
        bounds = [(0.0, 1.0), (0.01, 10.0), (0.0, 1.0), (0.01, 10.0),
                  (0.0, np.pi/2.0), (0.01, 20.0)]
    if options is None:
        options = {'maxiter': 100, 'disp': True}
    res = minimize(likelihood, p0,
                   args=(Psai_obs, eps_obs, size, bins),
                   bounds=bounds, method='L-BFGS-B', options=options)
    return res
