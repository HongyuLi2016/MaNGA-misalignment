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
# import emcee
# from JAM.utils import corner_plot
from JAM.utils import util_fig
# import matplotlib.pyplot as plt
# from time import time, localtime, strftime


util_fig.ticks_font.set_size(12)
util_fig.ticks_font1.set_size(8)
util_fig.label_font.set_size(25)


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


# def check_boundary(pars):
#     zeta, ksai, Psai_int = pars
#     if (boundary['zeta'][0] < zeta < boundary['zeta'][1]) and \
#             (boundary['ksai'][0] < ksai < zeta) and \
#             (boundary['Psai_int'][0] < Psai_int < boundary['Psai_int'][1]):
#         return True
#     else:
#         return False
#
#
# def lnprob_gaussian_A(pars, hypers={}):
#     '''
#     hypers -  a dictianry contains distribution parametetrs (i.e. means and
#               icov for gaussian)
#     '''
#     means = hypers['means']
#     icov = hypers['icov']
#     in_boundary = check_boundary(pars)
#     if not in_boundary:
#         return -np.inf
#     # zeta, ksai, Psai_int = pars
#     diff = pars - means
#     lnprob = -0.5 * np.dot(diff, np.dot(icov, diff))
#     return lnprob
#
#
# def flat_initp(keys, nwalkers):
#     '''
#     create initital positions for mcmc. Flat distribution within prior.
#     keys: List of parameter name
#     nwalkers: number of emcee walkers
#     '''
#     ndim = len(keys)
#     p0 = np.zeros([nwalkers, ndim])
#     for i in range(ndim):
#         p0[:, i] = np.random.uniform(low=init_boundary[keys[i]][0]+1e-4,
#                                      high=init_boundary[keys[i]][1]-1e-4,
#                                      size=nwalkers)
#     return p0
#
#
# def get_sample_mcmc_A(paras={}, nwalkers=300, burnin=300):
#     '''
#     Draw random an sample of b/a, c/a, Psai_int
#     '''
#     if 'method' not in paras.keys():
#         paras['method'] = 'gaussian'
#     if paras['method'] == 'gaussian':
#         lnprob = lnprob_gaussian_A
#     else:
#         raise KeyError('distribution function is invalid - check input method')
#     steps = paras.pop('steps', 1000)
#     nwalkers = paras.pop('nwalkers', 300)
#     burnin = paras.pop('burnin', 300)
#     threads = paras.pop('threads', 1)
#     # print information
#     date = strftime('%Y-%m-%d %X', localtime())
#     print('**************************************************')
#     startTime = time()
#     print('Toy distribution created at {}'.format(date))
#     print('method (distribution fuction): {}'.format(paras['method']))
#     print('nsteps: {}    nwalkers: {}'.format(steps, nwalkers))
#     print('burnin steps: {}'.format(burnin))
#     print('boundaries:')
#     print('zeta: [{:.2f}, {:.2f}]'.format(boundary['zeta'][0],
#                                           boundary['zeta'][1]))
#     print('ksai: [{:.2f}, {:.2f}]'.format(boundary['ksai'][0],
#                                           boundary['ksai'][1]))
#     print('Psai_int: [{:.2f}, {:.2f}]'.format(boundary['Psai_int'][0],
#                                               boundary['Psai_int'][1]))
#     print('Hyper parameters:')
#     print('--------------------')
#     for key in paras['hypers'].keys():
#         print(key)
#         print(paras['hypers'][key])
#         print('--------------------')
#     keys = ['zeta', 'ksai', 'Psai_int']
#     p0 = flat_initp(keys, nwalkers)
#     sampler = \
#         emcee.EnsembleSampler(nwalkers, 3, lnprob,
#                               kwargs={'hypers': paras['hypers']},
#                               threads=threads)
#     pos, prob, state = sampler.run_mcmc(p0, burnin)
#     sampler.reset()
#     sampler.run_mcmc(pos, steps)
#     rst = {}
#     rst['chain'] = sampler.chain
#     rst['lnprobability'] = sampler.lnprobability
#     rst['acceptance_fraction'] = sampler.acceptance_fraction
#     print('Finish! Total elapsed time for creation is: {:.2f}s'
#           .format(time()-startTime))
#     return rst
#
#
# def get_sample_mcmc_B(paras={}, nwalkers=300, burnin=300):
#     '''
#     Draw random an sample of b/a, c/b, Psai_int
#     '''
#     if 'method' not in paras.keys():
#         paras['method'] = 'gaussian'
#     if paras['method'] == 'gaussian':
#         lnprob = lnprob_gaussian_B
#     else:
#         raise KeyError('distribution function is invalid - check input method')
#     steps = paras.pop('steps', 1000)
#     nwalkers = paras.pop('nwalkers', 300)
#     burnin = paras.pop('burnin', 300)
#     threads = paras.pop('threads', 1)
#     # print information
#     date = strftime('%Y-%m-%d %X', localtime())
#     print('**************************************************')
#     startTime = time()
#     print('Toy distribution created at {}'.format(date))
#     print('method (distribution fuction): {}'.format(paras['method']))
#     print('nsteps: {}    nwalkers: {}'.format(steps, nwalkers))
#     print('burnin steps: {}'.format(burnin))
#     print('boundaries:')
#     print('zeta: [{:.2f}, {:.2f}]'.format(boundary['zeta'][0],
#                                           boundary['zeta'][1]))
#     print('ksai: [{:.2f}, {:.2f}]'.format(boundary['ksai'][0],
#                                           boundary['ksai'][1]))
#     print('Psai_int: [{:.2f}, {:.2f}]'.format(boundary['Psai_int'][0],
#                                               boundary['Psai_int'][1]))
#     print('Hyper parameters:')
#     print('--------------------')
#     for key in paras['hypers'].keys():
#         print(key)
#         print(paras['hypers'][key])
#         print('--------------------')
#     keys = ['zeta', 'ksai', 'Psai_int']
#     p0 = flat_initp(keys, nwalkers)
#     sampler = \
#         emcee.EnsembleSampler(nwalkers, 3, lnprob,
#                               kwargs={'hypers': paras['hypers']},
#                               threads=threads)
#     pos, prob, state = sampler.run_mcmc(p0, burnin)
#     sampler.reset()
#     sampler.run_mcmc(pos, steps)
#     rst = {}
#     rst['chain'] = sampler.chain
#     rst['lnprobability'] = sampler.lnprobability
#     rst['acceptance_fraction'] = sampler.acceptance_fraction
#     print('Finish! Total elapsed time for creation is: {:.2f}s'
#           .format(time()-startTime))
#     return rst
#
#
# def get_sample():
#     return
#
#
# def plot_chains(chain):
#     ndim = 3
#     labels = [r'$\mathbf{\zeta}$', r'$\mathbf{\xi}$',
#               r'$\mathbf{\Psi_{int}}$']
#     lim = [[0.0, 1.0], [0.0, 1.0], [0.0, np.pi/2.0]]
#     figsize = (8.0, ndim*2.0)
#     fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=figsize)
#     for i in range(ndim):
#         axes[i].plot(chain[:, :, i].T, color='k', alpha=0.01)
#         axes[i].set_ylim(lim[i])
#         axes[i].set_ylabel(labels[i])
#     axes[-1].set_xlabel('nstep')
#     return fig
#
#
# def analysis_sample(rst, figname='mcmc.png', outpath='.',
#                     clevel=[0.4, 0.683, 0.95, 0.997], truths=None,
#                     hbins=30, color=[0.8936, 0.5106, 0.2553, 0.01276],
#                     burnin=0, alpha=0.02, **kwargs):
#     labels = [r'$\mathbf{\zeta}$', r'$\mathbf{\xi}$',
#               r'$\mathbf{\Psi_{int}}$']
#     chain = rst['chain']
#     # fig_chain = plot_chains(rst['chain'])
#     # fig_chain.savefig('{}/chains.png'.format(outpath), dpi=100)
#     flatchain = rst['chain'][:, burnin:, :].reshape(-1, 3)
#     extents = [[0.0, 1.0], [0.0, 1.0], [0.0, np.pi/2.0]]
#     fig = corner_plot.corner(flatchain, clevel=clevel, hbins=hbins,
#                              truths=truths, color=color, resample=False,
#                              quantiles=[0.16, 0.5, 0.84], linewidths=2.0,
#                              labels=labels, extents=extents, **kwargs)
#     xpos = 0.53
#     ypos = 0.73
#     xsize = 0.45
#     ysize = 0.08
#     ystep = 0.01
#     axes0a = fig.add_axes([xpos, ypos, xsize, ysize])
#     axes0b = fig.add_axes([xpos, ypos+ysize+ystep, xsize, ysize])
#     axes0c = fig.add_axes([xpos, ypos+2.0*(ysize+ystep), xsize, ysize])
#     axes = [axes0c, axes0b, axes0a]
#     for i in range(3):
#         axes[i].plot(chain[:, :, i].T, color='k', alpha=alpha)
#         axes[i].set_ylim(extents[i])
#         axes[i].set_ylabel(labels[i])
#         util_fig.set_labels(axes[i], font=util_fig.ticks_font1)
#     axes[0].set_xticklabels([''])
#     axes[1].set_xticklabels([''])
#     axes[2].set_xlabel('nstep', fontproperties=util_fig.ticks_font)
#     fig.savefig('{}/{}'.format(outpath, figname), dpi=100)
#
#
# def analysis_distribution(rst, figname='joint_distribution.png', outpath='.',
#                           clevel=[0.4, 0.683, 0.95, 0.997],
#                           truths=None, hbins=30,
#                           color=[0.8936, 0.5106, 0.2553, 0.01276],
#                           burnin=0, alpha=0.05, **kwargs):
#     labels = [r'$\mathbf{T}$', r'$\mathbf{\Psi_{int}}$',
#               r'$\mathbf{\Psi}$', r'$\mathbf{\epsilon}$',
#               r'$\mathbf{cos\theta}$', r'$\mathbf{\phi}$']
#     # prepare for corner plot
#     T = (1-rst['zeta']**2)/(1-rst['ksai']**2)
#     data = np.zeros([len(rst['zeta']), 6])
#     data[:, 0] = T
#     data[:, 1] = rst['Psai_int']
#     data[:, 2] = rst['Psai']
#     data[:, 3] = rst['eps']
#     data[:, 4] = rst['costheta']
#     data[:, 5] = rst['phi']
#     extents = [[0.0, 1.0], [0.0, np.pi/2.0], [0.0, np.pi/2.0], [0.0, 1.0],
#                [0.0, 1.0], [0.0, 2.0*np.pi]]
#     fig = corner_plot.corner(data, clevel=clevel, hbins=hbins,
#                              truths=truths, color=color, resample=False,
#                              quantiles=[0.16, 0.5, 0.84], linewidths=2.0,
#                              labels=labels, extents=extents, **kwargs)
#     xpos = 0.53
#     ypos = 0.73
#     xsize = 0.45
#     ysize = 0.25
#     axes0a = fig.add_axes([xpos, ypos, xsize, ysize])
#     axes0a.plot(rst['eps'], np.degrees(rst['Psai']),
#                 '.k', alpha=alpha, markersize=1.0)
#     util_fig.set_labels(axes0a)
#     axes0a.set_xlim([0.0, 1.0])
#     axes0a.set_ylim([0.0, 90.0])
#     axes0a.set_xlabel('$\mathbf{\epsilon}$',
#                       fontproperties=util_fig.label_font)
#     axes0a.set_ylabel('$\mathbf{\Psi}$',
#                       fontproperties=util_fig.label_font)
#     fig.savefig('{}/{}'.format(outpath, figname), dpi=100)
