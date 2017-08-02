#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: util_sample.py
# Author: Hongyu Li <lhy88562189@gmail.com>
# Date: 02.08.2017
# Last Modified: 02.08.2017
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
# from scipy import stats
import emcee
from JAM.utils import corner_plot
from JAM.utils import util_fig
import matplotlib.pyplot as plt
from time import time, localtime, strftime
boundary = {'zeta': [0.5, 1.0], 'eta': [0.5, 1.0],
            'Psai_int': [0.0, np.pi/2.0]}


def check_boundary(pars):
    zeta, eta, Psai_int = pars
    if (boundary['zeta'][0] < zeta < boundary['zeta'][1]) and \
            (boundary['eta'][0] < eta < boundary['eta'][1]) and \
            (boundary['Psai_int'][0] < Psai_int < boundary['Psai_int'][1]):
        return True
    else:
        return False


def lnprob_gaussian(pars, hypers={}):
    '''
    hypers -  a dictianry contains distribution parametetrs (i.e. means and
              icov for gaussian)
    '''
    means = hypers['means']
    icov = hypers['icov']
    in_boundary = check_boundary(pars)
    if not in_boundary:
        return -np.inf
    # zeta, eta, Psai_int = pars
    diff = pars - means
    lnprob = -0.5 * np.dot(diff, np.dot(icov, diff))
    return lnprob


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


def get_sample_mcmc(paras={}, nwalkers=300, burnin=300):
    '''
    Draw random an sample of b/a, c/b, Psai_int
    '''
    if 'method' not in paras.keys():
        paras['method'] = 'gaussian'
    if paras['method'] == 'gaussian':
        lnprob = lnprob_gaussian
    else:
        raise KeyError('distribution function is invalid - check input method')
    steps = paras.pop('steps', 1000)
    nwalkers = paras.pop('nwalkers', 300)
    burnin = paras.pop('burnin', 300)
    threads = paras.pop('threads', 1)
    # print information
    date = strftime('%Y-%m-%d %X', localtime())
    print('**************************************************')
    startTime = time()
    print('Toy distribution created at {}'.format(date))
    print('method (distribution fuction): {}'.format(paras['method']))
    print('nsteps: {}    nwalkers: {}'.format(steps, nwalkers))
    print('burnin steps: {}'.format(burnin))
    print('boundaries:')
    print('zeta: [{:.2f}, {:.2f}]'.format(boundary['zeta'][0],
                                          boundary['zeta'][1]))
    print('eta: [{:.2f}, {:.2f}]'.format(boundary['eta'][0],
                                         boundary['eta'][1]))
    print('Psai_int: [{:.2f}, {:.2f}]'.format(boundary['Psai_int'][0],
                                              boundary['Psai_int'][1]))
    print('Hyper parameters:')
    print('--------------------')
    for key in paras['hypers'].keys():
        print(key)
        print(paras['hypers'][key])
        print('--------------------')
    keys = ['zeta', 'eta', 'Psai_int']
    p0 = flat_initp(keys, nwalkers)
    sampler = \
        emcee.EnsembleSampler(nwalkers, 3, lnprob,
                              kwargs={'hypers': paras['hypers']},
                              threads=threads)
    pos, prob, state = sampler.run_mcmc(p0, burnin)
    sampler.reset()
    sampler.run_mcmc(pos, steps)
    rst = {}
    rst['paras'] = keys
    rst['chain'] = sampler.chain
    rst['lnprobability'] = sampler.lnprobability
    rst['acceptance_fraction'] = sampler.acceptance_fraction
    print('Finish! Total elapsed time for creation is: {:.2f}s'
          .format(time()-startTime))
    return rst


def get_sample():
    return


def plot_chains(chain):
    ndim = 3
    labels = [r'$\mathbf{\zeta}$', r'$\mathbf{\eta}$',
              r'$\mathbf{\Psi_{int}}$']
    lim = [[0.0, 1.0], [0.0, 1.0], [0.0, np.pi/2.0]]
    figsize = (8.0, ndim*2.0)
    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=figsize)
    for i in range(ndim):
        axes[i].plot(chain[:, :, i].T, color='k', alpha=0.01)
        axes[i].set_ylim(lim[i])
        axes[i].set_ylabel(labels[i])
    axes[-1].set_xlabel('nstep')
    return fig


def analysis_sample(rst, figname='mcmc.png', outpath='.',
                    clevel=[0.4, 0.683, 0.95, 0.997], truths=None,
                    hbins=30, color=[0.8936, 0.5106, 0.2553, 0.01276],
                    burnin=0, alpha=0.02, **kwargs):
    labels = [r'$\mathbf{\zeta}$', r'$\mathbf{\eta}$',
              r'$\mathbf{\Psi_{int}}$']
    chain = rst['chain']
    # fig_chain = plot_chains(rst['chain'])
    # fig_chain.savefig('{}/chains.png'.format(outpath), dpi=100)
    flatchain = rst['chain'][:, burnin:, :].reshape(-1, 3)
    extents = [[0.0, 1.0], [0.0, 1.0], [0.0, np.pi/2.0]]
    fig = corner_plot.corner(flatchain, clevel=clevel, hbins=hbins,
                             truths=truths, color=color, resample=False,
                             quantiles=[0.16, 0.5, 0.84], linewidths=2.0,
                             labels=labels, extents=extents, **kwargs)
    xpos = 0.53
    ypos = 0.73
    xsize = 0.45
    ysize = 0.08
    ystep = 0.01
    axes0a = fig.add_axes([xpos, ypos, xsize, ysize])
    axes0b = fig.add_axes([xpos, ypos+ysize+ystep, xsize, ysize])
    axes0c = fig.add_axes([xpos, ypos+2.0*(ysize+ystep), xsize, ysize])
    axes = [axes0c, axes0b, axes0a]
    for i in range(3):
        axes[i].plot(chain[:, :, i].T, color='k', alpha=alpha)
        axes[i].set_ylim(extents[i])
        axes[i].set_ylabel(labels[i])
        util_fig.set_labels(axes[i], font=util_fig.ticks_font1)
    axes[0].set_xticklabels([''])
    axes[1].set_xticklabels([''])
    axes[2].set_xlabel('nstep', fontproperties=util_fig.ticks_font)
    fig.savefig('{}/{}'.format(outpath, figname), dpi=100)
