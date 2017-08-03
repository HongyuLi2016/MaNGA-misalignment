#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: util_hierarchical.py
# Author: Hongyu Li <lhy88562189@gmail.com>
# Date: 03.08.2017
# Last Modified: 03.08.2017
# ============================================================================
#  DESCRIPTION: ---
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
# ORGANIZATION:
#      VERSION: 0.0
# ============================================================================
import emcee
import numpy as np
import util_angle
from time import time, localtime, strftime
import socket

# import matplotlib.pyplot as plt
boundary = {'zeta': [0.5, 1.0], 'eta': [0.5, 1.0],
            'Psai_int': [0.0, np.pi/2.0], 'costheta': [-1.0, 1.0],
            'phi': [0.0, 2.0*np.pi]}
paraNames = ['zeta', 'eta', 'Psai_int', 'costheta', 'phi']
parLabels = [r'$\mathbf{\zeta}$', r'$\mathbf{\eta}$',
             r'$\mathbf{\Psi_{int}}$', r'$\mathbf{cos\theta}$',
             r'$\mathbf{\phi}$']


def check_boundary(pars):
    for i in range(len(pars)):
        if (boundary[paraNames[i]][0] < pars[i] < boundary[paraNames[i]][1]):
            pass
        else:
            return False
    return True


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


def lnprob(pars, Psai_obs=None, Psai_err=None, eps_obs=None, eps_err=None):
    in_boundary = check_boundary(pars)
    if not in_boundary:
        return -np.inf
    zeta, eta, Psai_int, costheta, phi = pars
    ksai = zeta * eta
    theta = np.arccos(costheta)
    Psai, eps = util_angle.get_Psai(theta, phi, zeta, ksai, Psai_int)
    lnprob_value = -0.5 * (((Psai[0]-Psai_obs)/Psai_err)**2 +
                           ((eps[0]-eps_obs)/eps_err)**2)
    return lnprob_value


def single_sample(Psai, eps, Psai_err=0.174, eps_err=0.05, nstep=1000,
                  burnin=500, nwalkers=200, ndim=5, threads=1):
    date = strftime('%Y-%m-%d %X', localtime())
    uname = socket.gethostname()
    print('**************************************************')
    startTime = time()
    print('Chain created at {} on {}'.format(date, uname))
    print('nstep: {}    nwalkers: {}    threads: {}'
          .format(nstep, nwalkers, threads))
    print('burnin steps: {}'.format(burnin))
    print('boundaries:')
    print('zeta: [{:.2f}, {:.2f}]'.format(boundary['zeta'][0],
                                          boundary['zeta'][1]))
    print('eta: [{:.2f}, {:.2f}]'.format(boundary['eta'][0],
                                         boundary['eta'][1]))
    print('Psai_int: [{:.2f}, {:.2f}]'.format(boundary['Psai_int'][0],
                                              boundary['Psai_int'][1]))
    print('Psai_obs: {:.1f}  err: {:.1f}'
          .format(np.degrees(Psai), np.degrees(Psai_err)))
    print('eps_obs: {:.2f}  err: {:.2f}'.format(eps, eps_err))
    p0 = flat_initp(paraNames, nwalkers)
    sampler = \
        emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                              kwargs={'Psai_obs': Psai, 'eps_obs': eps,
                                      'Psai_err': Psai_err,
                                      'eps_err': eps_err},
                              threads=threads)
    pos, prob, state = sampler.run_mcmc(p0, burnin)
    sampler.reset()
    sampler.run_mcmc(pos, nstep)
    print('Finish! Total elapsed time is: {:.2f}s'
          .format(time()-startTime))
    return sampler
