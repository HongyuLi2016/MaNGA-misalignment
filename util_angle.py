#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: util_angle.py
# Author: Hongyu Li <lhy88562189@gmail.com>
# Date: 28.07.2017
# Last Modified: 09.08.2017
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
    q = np.sqrt((A + C - sqrt_ABC) /
                (A + C + sqrt_ABC))
    return Gamma_min, q


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
    Gamma_min, q = get_Gamma_min(theta, phi, zeta, ksai)
    Gamma_kin = get_Gamma_kin(theta, phi, Psai_int)
    Psai = abs(Gamma_min - Gamma_kin)
    ii = Psai > np.pi/2.0
    Psai[ii] = np.pi - Psai[ii]
    return Psai, 1.0-q


def get_view_angle(size, seed=None):
    '''
    draw a random sample of viewing angles
    return
    theta - (0, pi)
    phi - (0, 2pi)
    '''
    if seed is None:
        seed_phi = None
        seed_theta = None
    else:
        seed_phi = seed % 863 + 11
        seed_theta = seed % 865 + 13
        if seed_phi == seed_theta:
            raise ValueError('seed phi and theta are the same!')
    phi = stats.uniform.rvs(size=size, loc=0.0, scale=2.0*np.pi,
                            random_state=seed_phi)
    cos_theta = stats.uniform.rvs(size=size, loc=-1.0, scale=2.0,
                                  random_state=seed_theta)
    theta = np.arccos(cos_theta)
    return theta, phi
