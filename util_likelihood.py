#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : util_likelihood.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 08.09.2017
# Last Modified Date: 08.09.2017
# Last Modified By  : Hongyu Li <lhy88562189@gmail.com>
# -*- coding: utf-8 -*-
# File              : util_likelihood.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 03.08.2017
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
import util_sample_A
from time import time, localtime, strftime
import socket
import sys
from scipy.interpolate import RectBivariateSpline
# import matplotlib.pyplot as plt

# ---------------------------- hyper mcmc A --------------------------------
boundary_A = {'mu_zeta': [0.5, 1.0], 'sigma_zeta': [0.03, 1.0],
              'mu_eta': [0.5, 1.0], 'sigma_eta': [0.03, 1.0],
              'mu_Psai_int': [0.0, np.pi/2.0],
              'sigma_Psai_int': [0.03, 1.5*np.pi]}
paraNames_A = ['mu_zeta', 'sigma_zeta', 'mu_eta', 'sigma_eta',
               'mu_Psai_int', 'sigma_Psai_int']
parLabels_A = [r'$\mathbf{\mu_{\zeta}}$', r'$\mathbf{\sigma_{\zeta}}$',
               r'$\mathbf{\mu_{\eta}}$', r'$\mathbf{\sigma_{\eta}}$',
               r'$\mathbf{\mu_{\Psi_{int}}}$',
               r'$\mathbf{\sigma_{\Psi_{int}}}$']


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


# ----------------------- hyper parameter sampler A --------------------------
def likelihood_A(pars, Psai_obs=None, eps_obs=None,
                 size=5000000, bins=25, seed=None,
                 theta=None, phi=None, interp=True):
    in_boundary = check_boundary(pars, boundary=boundary_A,
                                 paraNames=paraNames_A)
    if not in_boundary:
        return -np.inf
    mu_zeta, sigma_zeta, mu_eta, sigma_eta, mu_Psai_int, sigma_Psai_int = pars
    if theta is None:
        theta, phi = util_angle.get_view_angle(size, seed=seed)
    seed_zeta = (seed % 473) + 1 if seed is not None else None
    zeta = util_sample_A.get_sample(mu_zeta, sigma_zeta, boundary=[0.6, 1.0],
                                    size=size, seed=seed_zeta)
    seed_eta = (seed % 473) + 3 if seed is not None else None
    eta = util_sample_A.get_sample(mu_eta, sigma_eta, boundary=[0.6, 1.0],
                                   size=size, seed=seed_eta)
    ksai = eta * zeta
    seed_Psai_int = (seed % 473) + 5 if seed is not None else None
    Psai_int = util_sample_A.get_sample(mu_Psai_int, sigma_Psai_int,
                                        boundary=[0.0, np.pi/2.0],
                                        size=size, seed=seed_Psai_int)
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
    if np.isnan(lnprob):
        return -np.inf
    # print pars, lnprob
    return lnprob


def hyperMCMC_A(eps_obs, Psai_obs, nstep=1000, burnin=500, nwalkers=200,
                ndim=6, threads=1, size=5000000, bins=25, seed=None,
                interp=True):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    proc_size = comm.Get_size()
    theta, phi = util_angle.get_view_angle(size, seed=seed)
    if rank == 0:
        date = strftime('%Y-%m-%d %X', localtime())
        uname = socket.gethostname()
        print('**************************************************')
        startTime = time()
        print('hyperMCMC for model A run at {} on {}'.format(date, uname))
        print('nstep: {}    nwalkers: {}    nprocesses: {}'
              .format(nstep, nwalkers, proc_size))
        print('number of galaxies: {}'.format(len(eps_obs)))
        print('number of random points: {}'.format(size))
        print('number of bins: {}'.format(bins))
        print('interpolation: {}'.format(interp))
        print('random number seed: {}'.format(seed))
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
        emcee.EnsembleSampler(nwalkers, ndim, likelihood_A,
                              kwargs={'eps_obs': eps_obs,
                                      'Psai_obs': Psai_obs,
                                      'size': size, 'bins': bins,
                                      'seed': seed, 'theta': theta,
                                      'phi': phi, 'interp': interp},
                              pool=pool)

    # ------------------------ burnin --------------------------
    print_burnin = burnin // 20
    print 'burning start'
    sys.stdout.flush()
    for i, result in enumerate(sampler.sample(p0, iterations=burnin)):
        if ((i+1) % print_burnin == 0) and (rank == 0):
            print("{0:5.1%}".format(float(i+1) / burnin))
            sys.stdout.flush()
    print 'burning finish'
    # pos, prob, state = sampler.run_mcmc(p0, burnin)
    pos, prob, state = result
    sampler.reset()
    # ---------------------- run ----------------------------
    print_run = nstep // 20
    print 'run start'
    for i, result in enumerate(sampler.sample(result[0], iterations=nstep)):
        if ((i+1) % print_run == 0) and (rank == 0):
            print("{0:5.1%}".format(float(i+1) / nstep))
            sys.stdout.flush()
    print 'finish'

    # sampler.run_mcmc(pos, nstep)
    pool.close()
    print('Finish! Total elapsed time is: {:.2f}s'
          .format(time()-startTime))
    return sampler
