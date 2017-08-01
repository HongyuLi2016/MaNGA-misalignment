#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: util_angle.py
# Author: Hongyu Li <lhy88562189@gmail.com>
# Date: 28.07.2017
# Last Modified: 31.07.2017
# ============================================================================
#  DESCRIPTION: MC simulation for MaNGA prolate project
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
# ORGANIZATION:
#      VERSION: 0.0
# ============================================================================
import numpy as np
from scipy import stats
import emcee
boundary = {'zeta': [0.5, 1.0], 'ksai': [0.5, 1.0], 'Psai_int': [0.0, np.pi]}


def get_Gamma_kin(theta, phi, Psai_int):
    '''
    Calculate Gamma_kin using Eq. (6) in Franx et al. (1991)
    input theta, phi, Psai_int should be in radians
    theta [0, pi]
    phi [0, 2pi]
    Psai_int [0, pi/2], could be a scalar or an array with the same dimension
             as theta and phi
    output
    Gamma_kin [0, pi]
    '''
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_Psai_int = np.sin(Psai_int)
    cos_Psai_int = np.cos(Psai_int)
    tan_Gamma_kin = sin_phi*sin_Psai_int / (-cos_phi*cos_theta*sin_Psai_int +
                                            sin_theta*cos_Psai_int)
    Gamma_kin = np.arctan(tan_Gamma_kin)
    ii = Gamma_kin < 0.0
    Gamma_kin[ii] += np.pi
    return Gamma_kin


def get_Gamma_min(theta, phi, zeta, ksai):
    '''
    Calculate eps and Gamma_min using Eq. (12) and (13) in Binney (1985)
    input theta, phi should be in radians
    theta [0, pi]
    phi [0, 2pi]
    zeta b/a, could be a scalar or an array with the same dimension
         as theta and phi
    ksai c/a, could be a scalar or an array with the same dimension
         as theta and phi
    1 >= zeta >= ksai
    output
    Gamma_min [0, pi]
    '''
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_theta_square = sin_theta**2
    cos_theta_square = cos_theta**2
    sin_phi_square = sin_phi**2
    cos_phi_square = cos_phi**2
    sin_two_phi = np.sin(2.0*phi)

    zeta_square = zeta**2
    ksai_square = ksai**2

    A = cos_theta_square/ksai_square * \
        (sin_phi_square+cos_phi_square/zeta_square) + \
        sin_theta_square/zeta_square
    B = cos_theta * sin_two_phi * (1.0-1.0/zeta_square) / ksai_square
    C = (sin_phi_square/zeta_square + cos_phi_square) / ksai_square
    tan_two_Gamma_min = B/(A-C)
    two_Gamma_min = np.arctan(tan_two_Gamma_min)
    ii = two_Gamma_min < 0.0
    two_Gamma_min[ii] += np.pi
    sign = (A-C)*np.cos(two_Gamma_min) + B*np.sin(two_Gamma_min)
    Gamma_min = two_Gamma_min * 0.5
    ii = sign > 0.0
    Gamma_min[ii] += np.pi/2.0
    sqrt_ABC = np.sqrt((A-C)**2 + B**2)
    eps = np.sqrt((A + C - sqrt_ABC) /
                  (A + C + sqrt_ABC))
    return Gamma_min, eps


def get_Psai(theta, phi, zeta, ksai, Psai_int):
    '''
    Calculate misalignment angle using Eq. (13) in Binney (1985) and Eq. (6)
    in Franx et al. (1991)
    input theta, phi should be in radians
    theta [0, pi]
    phi [0, 2pi]
    Psai_int [0, pi/2], could be a scalar or an array with the same dimension
             as theta and phi
    zeta b/a, could be a scalar or an array with the same dimension
         as theta and phi
    ksai c/a, could be a scalar or an array with the same dimension
         as theta and phi
    1 >= zeta >= ksai
    output
    Gamma_min [0, pi]
    '''
    Gamma_min, eps = get_Gamma_min(theta, phi, zeta, ksai)
    Gamma_kin = get_Gamma_kin(theta, phi, Psai_int)
    Psai = abs(Gamma_min - Gamma_kin)
    ii = Psai > np.pi/2.0
    Psai[ii] = np.pi - Psai[ii]
    return Psai, eps


def get_view_angle(size):
    '''
    draw a random sample of viewing angles
    return
    theta - (0, pi)
    phi - (0, 2pi)
    '''
    phi = stats.uniform.rvs(size=size, loc=0.0, scale=2.0*np.pi)
    cos_theta = stats.uniform.rvs(size=size, loc=-1.0, scale=2.0)
    theta = np.arccos(cos_theta)
    return theta, phi


def check_boundary(pars):
    zeta, ksai, Psai_int = pars
    if (boundary['zeta'][0] < zeta < boundary['zeta'][1]) and \
            (boundary['ksai'][0] < ksai < zeta) and \
            (boundary['Psai_int'][0] < Psai_int < boundary['Psai_int'][1]):
        return 0.0
    else:
        return -np.inf


def lnprob_gaussian(pars, means=None, icov=None):
    boundary = check_boundary(pars)
    if np.isinf(boundary):
        return boundary
    # zeta, ksai, Psai_int = pars
    diff = pars - means
    return -0.5 * np.dot(diff, np.dot(icov, diff))


def flat_initp(keys, nwalkers):
    '''
    create initital positions for mcmc. Flat distribution within prior.
    keys: List of parameter name
    nwalkers: number of emcee walkers
    '''
    ndim = len(keys)
    p0 = np.zeros([nwalkers, ndim])
    for i in range(ndim):
        p0[:, i] = np.random.uniform(low=boundary[keys[i]][0]+1e-4,
                                     high=boundary[keys[i]][1]-1e-4,
                                     size=nwalkers)
    return p0


def get_sample_mcmc(size, method='gaussian', nwalkers=300, burnin=300):
    if method == 'gaussian':
        lnprob = lnprob_gaussian
    keys = ['zeta', 'ksai', 'Psai_int']
    p0 = flat_initp(keys, nwalkers)
    return


def get_sample():
    return
