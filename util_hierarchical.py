#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: util_hierarchical.py
# Author: Hongyu Li <lhy88562189@gmail.com>
# Date: 03.08.2017
# Last Modified: 05.08.2017
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
from emcee.utils import MPIPool
from mpi4py import MPI
import numpy as np
import util_angle
from time import time, localtime, strftime
import socket
import sys

# import matplotlib.pyplot as plt
# ----------------------------- single mcmc -------------------------------
boundary = {'zeta': [0.5, 1.0], 'eta': [0.5, 1.0],
            'Psai_int': [0.0, np.pi/2.0], 'costheta': [-1.0, 1.0],
            'phi': [0.0, 2.0*np.pi]}
paraNames = ['zeta', 'eta', 'Psai_int', 'costheta', 'phi']
parLabels = [r'$\mathbf{\zeta}$', r'$\mathbf{\eta}$',
             r'$\mathbf{\Psi_{int}}$', r'$\mathbf{cos\theta}$',
             r'$\mathbf{\phi}$']

# ---------------------------- hyper mcmc A --------------------------------
boundary_A = {'mu_zeta': [0.5, 1.0], 'sigma_zeta': [0.0, 10.0],
              'mu_eta': [0.5, 1.0], 'sigma_eta': [0.0, 10.0],
              'mu_Psai_int': [0.0, np.pi/2.0],
              'sigma_Psai_int': [0.0, 10.0*np.pi]}
paraNames_A = ['mu_zeta', 'sigma_zeta', 'mu_eta', 'sigma_eta',
               'mu_Psai_int', 'sigma_Psai_int']
parLabels_A = [r'$\mathbf{\mu_{\zeta}}$', r'$\mathbf{\sigma_{\zeta}}$',
               r'$\mathbf{\mu_{\eta}}$', r'$\mathbf{\sigma_{\eta}}$',
               r'$\mathbf{\mu_{\Psi_{int}}}$',
               r'$\mathbf{\sigma_{\Psi_{int}}}$']


# read in chains from single galaxies
def create_list(fname, n=1):
    glist = np.genfromtxt(fname, dtype='30S')
    paras_list = [np.load('{}/chain.npy'.format(gname))[::n, 0:3]
                  for gname in glist]
    return paras_list


# ---------------------- single mcmc for (eps, Psai) -------------------------
def check_boundary(pars, boundary=None, paraNames=None):
    for i in range(len(pars)):
        if (boundary[paraNames[i]][0] < pars[i] < boundary[paraNames[i]][1]):
            pass
        else:
            return False
    return True


def flat_initp(keys, nwalkers, boundary=None):
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
    in_boundary = check_boundary(pars, boundary=boundary, paraNames=paraNames)
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
    for key in paraNames:
        print('{}: [{:.2f}, {:.2f}]'.format(key, boundary[key][0],
                                            boundary[key][1]))
    p0 = flat_initp(paraNames, nwalkers, boundary=boundary)
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
# ------------------------- single sampler end -------------------------------


# ----------------------- hyper parameter sampler A --------------------------
def likelihood_A_i(pars, hyper_pars=None):
    '''
    likelihood of a single observation (eps, Psai for a single galaxies)
    hyper_pars:
        [mu_zeta, sigma_zeta, mu_eta, sigma_eta, mu_Psai_int, sigma_Psai_int]
    pars: MCMC chain from single sampler for a given (eps, Psai).  N*3 array
    '''
    mu = hyper_pars[[0, 2, 4]]
    sigma = hyper_pars[[1, 3, 5]]
    x = np.sum(((pars-mu)/sigma)**2, axis=1)
    lnlikelihood = np.log(np.sum(np.exp(-0.5*x)))
    return lnlikelihood


def lnprob_hyper_A(hyper_pars, pars_list=None):
    in_boundary = check_boundary(hyper_pars, boundary=boundary_A,
                                 paraNames=paraNames_A)
    if not in_boundary:
        return -np.inf
    likelihood_list = \
        [likelihood_A_i(pars, hyper_pars=hyper_pars) for pars in pars_list]
    return np.sum(likelihood_list)


def hyperMCMC_A(pars_list, nstep=1000, burnin=500, nwalkers=200,
                ndim=6, threads=1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        date = strftime('%Y-%m-%d %X', localtime())
        uname = socket.gethostname()
        print('**************************************************')
        startTime = time()
        print('hyperMCMC for model A run at {} on {}'.format(date, uname))
        print('nstep: {}    nwalkers: {}    nprocesses: {}'
              .format(nstep, nwalkers, size))
        print('number of galaxies: {}'.format(len(pars_list)))
        print('Integration points: {}'.format(len(pars_list[0])))
        print('burnin steps: {}'.format(burnin))
        print('boundaries:')
        for key in paraNames_A:
            print('{}: [{:.2f}, {:.2f}]'.format(key, boundary_A[key][0],
                                                boundary_A[key][1]))
        sys.stdout.flush()
    p0 = flat_initp(paraNames_A, nwalkers, boundary_A)
    pool = MPIPool(loadbalance=True)
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    sampler = \
        emcee.EnsembleSampler(nwalkers, ndim, lnprob_hyper_A,
                              kwargs={'pars_list': pars_list},
                              pool=pool)
    pos, prob, state = sampler.run_mcmc(p0, burnin)
    sampler.reset()
    sampler.run_mcmc(pos, nstep)
    pool.close()
    print('Finish! Total elapsed time is: {:.2f}s'
          .format(time()-startTime))
    return sampler
