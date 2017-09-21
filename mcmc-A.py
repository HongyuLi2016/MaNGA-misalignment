#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : run_maxL-mcmc.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 11.09.2017
# Last Modified Date: 11.09.2017
# Last Modified By  : Hongyu Li <lhy88562189@gmail.com>
# ============================================================================
#  DESCRIPTION: ---
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
# ORGANIZATION:
#      VERSION: 0.0
# ============================================================================
# import util_maxL
import util_likelihood as ul
import numpy as np
# import util_analysis
# from JAM.utils import corner_plot
from optparse import OptionParser
import pickle
import os
import matplotlib.pyplot as plt


def plot_chains(chain):
    ndim = 6
    lim = [ul.boundary_A[key] for key in ul.paraNames_A]
    figsize = (8.0, ndim*2.0)
    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=figsize)
    for i in range(ndim):
        axes[i].plot(chain[:, :, i].T, color='k', alpha=0.1)
        axes[i].set_ylim(lim[i])
        axes[i].set_ylabel(ul.parLabels_A[i])
    axes[-1].set_xlabel('nstep')
    return fig


parser = OptionParser()
parser.add_option('-f', action='store', type='string', dest='fname',
                  default='SR-catalogue.dat', help='obs data file name')
(options, args) = parser.parse_args()
if len(args) != 1:
    raise KeyError('please provide a folder name')
try:
    os.mkdir(args[0])
except OSError:
    pass


data = np.genfromtxt(options.fname, usecols=[1, 3])
eps_obs = data[:, 1]
Psai_obs = np.radians(data[:, 0])
sampler = ul.hyperMCMC_A(eps_obs, Psai_obs, nstep=300, burnin=200,
                         nwalkers=200, size=5000000, bins=25, seed=88562189,
                         interp=True)

chain = sampler.chain
lnprobability = sampler.lnprobability
flatchain = chain.reshape(-1, 6)
flatlnprob = lnprobability.reshape(-1)

rst = {}
rst['parasName'] = ul.paraNames_A
rst['flatchain'] = flatchain
rst['flatlnprob'] = flatlnprob
with open('{}/hyper_chain.dat'.format(args[0]), 'w') as f:
    pickle.dump(rst, f)

fig = plot_chains(chain)
fig.savefig('{}/hyper_chain.png'.format(args[0]))

# clevel = [0.4, 0.683, 0.95, 0.997]
# color = [0.8936, 0.5106, 0.2553, 0.01276]
# hbins = 30
# lim = [ul.boundary_A[key] for key in ul.paraNames_A]
# fig = corner_plot.corner(flatchain, clevel=clevel, hbins=hbins,
#                          color=color, resample=False,
#                          quantiles=[0.16, 0.5, 0.84], linewidths=2.0,
#                          labels=ul.parLabels_A)
# fig.savefig('{}/hyper_mcmc.png'.format(args[0]))
