#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: util_analysis.py
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
from JAM.utils import corner_plot
from JAM.utils import util_fig


def analysis_distribution(rst, figname='joint_distribution.png', outpath='.',
                          clevel=[0.4, 0.683, 0.95, 0.997],
                          truths=None, hbins=30,
                          color=[0.8936, 0.5106, 0.2553, 0.01276],
                          burnin=0, alpha=0.05, **kwargs):
    labels = [r'$\mathbf{T}$', r'$\mathbf{\Psi_{int}}$',
              r'$\mathbf{\Psi}$', r'$\mathbf{\epsilon}$',
              r'$\mathbf{cos\theta}$', r'$\mathbf{\phi}$']
    # prepare for corner plot
    T = (1-rst['zeta']**2)/(1-rst['ksai']**2)
    data = np.zeros([len(rst['zeta']), 6])
    data[:, 0] = T
    data[:, 1] = rst['Psai_int']
    data[:, 2] = rst['Psai']
    data[:, 3] = rst['eps']
    data[:, 4] = rst['costheta']
    data[:, 5] = rst['phi']
    extents = [[0.0, 1.0], [0.0, np.pi/2.0], [0.0, np.pi/2.0], [0.0, 1.0],
               [-1.0, 1.0], [0.0, 2.0*np.pi]]
    fig = corner_plot.corner(data, clevel=clevel, hbins=hbins,
                             truths=truths, color=color, resample=False,
                             quantiles=[0.16, 0.5, 0.84], linewidths=2.0,
                             labels=labels, extents=extents, **kwargs)
    xpos = 0.53
    ypos = 0.73
    xsize = 0.45
    ysize = 0.25
    axes0a = fig.add_axes([xpos, ypos, xsize, ysize])
    axes0a.plot(rst['eps'], np.degrees(rst['Psai']),
                '.k', alpha=alpha, markersize=1.0)
    util_fig.set_labels(axes0a)
    axes0a.set_xlim([0.0, 1.0])
    axes0a.set_ylim([0.0, 90.0])
    axes0a.set_xlabel('$\mathbf{\epsilon}$',
                      fontproperties=util_fig.label_font)
    axes0a.set_ylabel('$\mathbf{\Psi}$',
                      fontproperties=util_fig.label_font)
    fig.savefig('{}/{}'.format(outpath, figname), dpi=100)
