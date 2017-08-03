#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test_hierarchical.py
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
import util_hierarchical as uh
import numpy as np
from JAM.utils import corner_plot
import matplotlib.pyplot as plt
from optparse import OptionParser
# from JAM.utils import util_fig


def plot_chains(chain):
    ndim = 5
    lim = [uh.boundary[key] for key in uh.paraNames]
    figsize = (8.0, ndim*2.0)
    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=figsize)
    for i in range(ndim):
        axes[i].plot(chain[:, :, i].T, color='k', alpha=0.01)
        axes[i].set_ylim(lim[i])
        axes[i].set_ylabel(uh.parLabels[i])
    axes[-1].set_xlabel('nstep')
    return fig


parser = OptionParser()
(options, args) = parser.parse_args()
if len(args) != 1:
    raise KeyError('please provide a folder name')
Psai, Psai_err, eps, eps_err = np.load('{}/data.npy'.format(args[0]))

sampler = uh.single_sample(np.radians(Psai), eps,
                           Psai_err=np.radians(Psai_err),
                           eps_err=eps_err, nwalkers=200,
                           nstep=3000, threads=1)
chain = sampler.chain
lnprobability = sampler.lnprobability
flatchain = chain.reshape(-1, 5)
flatlnprob = lnprobability.reshape(-1)
fig = plot_chains(chain)
fig.savefig('{}/chain.png'.format(args[0]))

clevel = [0.4, 0.683, 0.95, 0.997]
color = [0.8936, 0.5106, 0.2553, 0.01276]
hbins = 30
lim = [uh.boundary[key] for key in uh.paraNames]
fig = corner_plot.corner(flatchain, clevel=clevel, hbins=hbins,
                         color=color, resample=False,
                         quantiles=[0.16, 0.5, 0.84], linewidths=2.0,
                         labels=uh.parLabels, extents=lim)
fig.savefig('{}/mcmc.png'.format(args[0]))
rst = np.column_stack((flatchain, flatlnprob))
np.save('{}/chain.npy'.format(args[0]), rst)
