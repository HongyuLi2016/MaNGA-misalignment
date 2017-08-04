#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tem_test.py
# Author: Hongyu Li <lhy88562189@gmail.com>
# Date: 04.08.2017
# Last Modified: 04.08.2017
# ============================================================================
#  DESCRIPTION: ---
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
# ORGANIZATION:
#      VERSION: 0.0
# ============================================================================
# import numpy as np
import util_hierarchical as uh
from optparse import OptionParser
from JAM.utils import corner_plot
import matplotlib.pyplot as plt
import pickle
import os


def plot_chains(chain):
    ndim = 6
    lim = [uh.boundary_A[key] for key in uh.paraNames_A]
    figsize = (8.0, ndim*2.0)
    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=figsize)
    for i in range(ndim):
        axes[i].plot(chain[:, :, i].T, color='k', alpha=0.1)
        axes[i].set_ylim(lim[i])
        axes[i].set_ylabel(uh.parLabels_A[i])
    axes[-1].set_xlabel('nstep')
    return fig


parser = OptionParser()
parser.add_option('-f', action='store', type='string', dest='gname',
                  default='list.dat', help='galaxy name list')
(options, args) = parser.parse_args()
if len(args) != 1:
    raise KeyError('please provide a folder name')
try:
    os.mkdir(args[0])
except OSError:
    pass
para_list = uh.create_list(options.gname)

ndim = 6
sampler = uh.hyperMCMC_A(para_list, nstep=1000, burnin=500, ndim=ndim)
chain = sampler.chain
lnprobability = sampler.lnprobability
flatchain = chain.reshape(-1, ndim)
flatlnprob = lnprobability.reshape(-1)

rst = {}
rst['parasName'] = uh.paraNames_A
rst['flatchain'] = flatchain
rst['flatlnprob'] = flatlnprob
with open('{}/hyper_chain.dat'.format(args[0]), 'w') as f:
    pickle.dump(rst, f)

fig = plot_chains(chain)
fig.savefig('{}/hyper_chain.png'.format(args[0]))

clevel = [0.4, 0.683, 0.95, 0.997]
color = [0.8936, 0.5106, 0.2553, 0.01276]
hbins = 30
lim = [uh.boundary_A[key] for key in uh.paraNames_A]
fig = corner_plot.corner(flatchain, clevel=clevel, hbins=hbins,
                         color=color, resample=False,
                         quantiles=[0.16, 0.5, 0.84], linewidths=2.0,
                         labels=uh.parLabels_A, extents=lim)
fig.savefig('{}/hyper_mcmc.png'.format(args[0]))
