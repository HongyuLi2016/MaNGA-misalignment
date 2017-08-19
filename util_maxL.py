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
from scipy.interpolate import RectBivariateSpline


def likelihood(pars, Psai_obs=None, eps_obs=None,
               theta=None, phi=None,
               size=3000000, bins=50, seed=88562189,
               interp=True):
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
    if not interp:
        i_eps = (np.floor(eps_obs / (1.0 / bins))).astype(int)
        i_Psai = (np.floor(Psai_obs / (np.pi/2.0 / bins))).astype(int)
        lnprob = np.log(H[i_eps, i_Psai]).sum()
        # print pars, lnprob
    else:
        # plt.imshow(H, origin='lower')
        # plt.savefig('img.png')
        x_grid = 0.5 * (xedges[1:] + xedges[0:-1])
        y_grid = 0.5 * (yedges[1:] + yedges[0:-1])
        # xx = np.linspace(0, 1.0, 100)
        # yy = np.linspace(0, np.pi/2.0, 100)
        # X, Y = np.meshgrid(xx, yy, indexing='ij')
        # print X.shape, Y.shape
        # x: eps  y: Psai   H: row - eps  column - Psai
        f_lnprob = RectBivariateSpline(x_grid, y_grid, H, kx=1, ky=1,
                                       bbox=[0.0, 1.0, 0.0, np.pi/2.0])
        prob_obs = f_lnprob.ev(eps_obs, Psai_obs).clip(1e-8, None)
        lnprob = np.log(prob_obs).sum()
        # plt.imshow(lnprob_obs, origin='lower')
        # plt.savefig('img_interpo.png')
        # exit()
        print pars, lnprob
    if np.isnan(lnprob):
        return np.inf
    return -lnprob


'''
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
'''


def run_minimize(Psai_obs, eps_obs, p0, size=3000000, bins=50, bounds=None,
                 options=None, seed=88562189, interp=True):
    np.random.seed(seed=seed)
    theta, phi = util_angle.get_view_angle(size)
    if bounds is None:
        bounds = [(0.0, 1.0), (0.03, 4.0), (0.0, 1.0), (0.03, 4.0),
                  (0.0, np.pi/2.0), (0.03, 2.0*np.pi)]
    if options is None:
        options = {'maxiter': 1000, 'disp': True, 'eps': 0.0001}
    res = minimize(likelihood, p0,
                   args=(Psai_obs, eps_obs, theta, phi, size, bins,
                         seed, interp),
                   bounds=bounds, method='L-BFGS-B', options=options)
    return res
