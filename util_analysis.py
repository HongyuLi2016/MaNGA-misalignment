#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : util_analysis.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 08.09.2017
# Last Modified Date: 21.09.2017
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
import numpy as np
# from scipy import stats
from JAM.utils import corner_plot
from JAM.utils import util_fig
import matplotlib.pyplot as plt
np.seterr(all='ignore')


def hist2d(x, y, ax=None, xlim=[0.0, 1.0], ylim=[-90.0, 90.0], bins=100,
           cmap='Greys', log=False, per=[0.5, 99.5]):
    H, xedges, yedges = \
        np.histogram2d(x, y, range=[xlim, ylim],
                       normed=True, bins=bins)
    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    if log:
        img = np.log10(H.T)
    else:
        img = H.T
    vmin, vmax = np.nanpercentile(img, per)
    ax.imshow(img, origin='lower', extent=extent, aspect='auto', cmap=cmap,
              vmin=vmin, vmax=vmax)
    return


def analysis_distribution(rst, figname='joint_distribution.png', outpath='.',
                          clevel=[0.4, 0.683, 0.95, 0.997],
                          truths=None, hbins=30,
                          color=[0.8936, 0.5106, 0.2553, 0.01276],
                          burnin=0, alpha=0.05, **kwargs):
    labels = [r'$\mathbf{\zeta}$', r'$\mathbf{\eta}$',
              r'$\mathbf{\Psi_{int}}$',
              r'$\mathbf{\Psi}$', r'$\mathbf{\epsilon}$',
              r'$\mathbf{cos\theta}$', r'$\mathbf{\phi}$']
    # prepare for corner plot
    data = np.zeros([len(rst['zeta']), 7])
    data[:, 0] = rst['zeta']
    data[:, 1] = rst['eta']
    data[:, 2] = rst['Psai_int']
    data[:, 3] = rst['Psai']
    data[:, 4] = rst['eps']
    data[:, 5] = rst['costheta']
    data[:, 6] = rst['phi']
    extents = [[0.0, 1.0], [0.0, 1.0], [0.0, np.pi/2.0], [0.0, np.pi/2.0],
               [0.0, 1.0], [-1.0, 1.0], [0.0, 2.0*np.pi]]
    fig = corner_plot.corner(data, clevel=clevel, hbins=hbins,
                             truths=truths, color=color, resample=False,
                             quantiles=[0.16, 0.5, 0.84], linewidths=2.0,
                             labels=labels, extents=extents, **kwargs)
    xpos = 0.53
    ypos = 0.73
    xsize = 0.45
    ysize = 0.25
    axes0a = fig.add_axes([xpos, ypos, xsize, ysize])
    # axes0a.plot(rst['eps'], np.degrees(rst['Psai']),
    #             '.k', alpha=alpha, markersize=1.0)
    hist2d(rst['eps'], np.degrees(rst['Psai']), ax=axes0a,
           xlim=[0.0, 1.0], ylim=[0.0, 90.0], bins=100,
           log=False, cmap='afmhot_r')
    util_fig.set_labels(axes0a)
    axes0a.set_xlim([0.0, 1.0])
    axes0a.set_ylim([0.0, 90.0])
    axes0a.set_xlabel('$\mathbf{\epsilon}$',
                      fontproperties=util_fig.label_font)
    axes0a.set_ylabel('$\mathbf{\Psi}$',
                      fontproperties=util_fig.label_font)

    xpos = 0.75
    ypos = 0.45
    xsize = 0.20
    ysize = 0.20
    axes0b = fig.add_axes([xpos, ypos, xsize, ysize])
    # axes0b.plot(rst['zeta'], rst['ksai'],
    #             '.k', alpha=alpha, markersize=1.0)
    hist2d(rst['zeta'], rst['ksai'], ax=axes0b,
           xlim=[0.0, 1.0], ylim=[0.0, 1.0], bins=100,
           log=False, cmap='afmhot_r')
    x = np.linspace(0.0, 1.0, 500)
    for Tri in [0.1, 0.3, 0.5, 0.7, 0.9]:
        y = np.sqrt(1.0 - (1.0-x**2)/Tri)
        axes0b.plot(x, y, '--r', lw=2.0)
    util_fig.set_labels(axes0b)
    axes0b.set_xlim([0.0, 1.0])
    axes0b.set_ylim([0.0, 1.0])
    axes0b.plot([0.0, 1.0], [0.0, 1.0], '--r', lw=2.0)
    axes0b.set_xlabel('$\mathbf{\zeta}$',
                      fontproperties=util_fig.label_font)
    axes0b.set_ylabel(r'$\mathbf{\xi}$',
                      fontproperties=util_fig.label_font)
    fig.savefig('{}/{}'.format(outpath, figname), dpi=100)


def plot_axis_ratio(ba, ca, outpath='.', markersize=1.0, alpha=0.02):
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    fig.subplots_adjust(left=0.130, bottom=0.130, right=0.98,
                        top=0.98, wspace=0.1, hspace=0.26)
    axes.plot(ba, ca, '.k', markersize=markersize, alpha=alpha)
    lim = np.array([0.0, 1.0])
    axes.set_xlim(lim)
    axes.set_ylim(lim)
    axes.plot(lim, lim, '--r', lw=2.0)
    util_fig.set_labels(axes)
    axes.set_xlabel(r'$\mathbf{\zeta}$',
                    fontproperties=util_fig.label_font)
    axes.set_ylabel(r'$\mathbf{\xi}$',
                    fontproperties=util_fig.label_font)
    fig.savefig('{}/ba-ca.png'.format(outpath))


def cmp_obs(rst, figname='cmp_obs.png', outpath='.', symmetry=True,
            mks=3.0, bins=19, xlim=[0.0, 1.0], lw=2.0):
    # util_fig.text_font.set_size()
    util_fig.ticks_font.set_size(12)
    util_fig.ticks_font1.set_size(10)
    Psai_obs = np.degrees(np.pi/2.0 - rst['Psai_obs'])
    eps_obs = rst['eps_obs']
    Psai = np.degrees(np.pi/2.0 - rst['Psai'])
    eps = rst['eps']
    if symmetry:
        eps = np.hstack([eps, eps])
        eps_obs = np.hstack([eps_obs, eps_obs])
        Psai = np.hstack([Psai, -Psai])
        Psai_obs = np.hstack([Psai_obs, -Psai_obs])
        ylim = [-90.0, 90.0]
    else:
        ylim = [0.0, 90.0]

    fig = plt.figure(figsize=(8, 4))
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=2)
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    axes = [ax1, ax2, ax3]
    fig.subplots_adjust(left=0.10, bottom=0.14, right=0.98,
                        top=0.98, wspace=0.3, hspace=0.40)
    hist2d(eps, Psai, ax=axes[0],
           xlim=xlim, ylim=ylim, bins=100,
           log=False, cmap='afmhot_r')
    axes[0].plot(eps_obs, Psai_obs, 'ob', markersize=mks)
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].set_xlabel('$\mathbf{\epsilon}$',
                       fontproperties=util_fig.label_font)
    axes[0].set_ylabel('$\mathbf{\Delta PA}$',
                       fontproperties=util_fig.label_font)

    axes[1].hist(Psai_obs, histtype='step', color='r', lw=lw, bins=bins,
                 range=ylim, normed=True, cumulative=True)
    axes[1].hist(Psai, histtype='step', color='k', lw=lw, bins=bins,
                 range=ylim, normed=True, cumulative=True)
    axes[1].set_xlim(ylim)
    axes[1].set_ylabel('cumulative',
                       fontproperties=util_fig.label_font)
    axes[1].set_xlabel('$\mathbf{\Delta PA}$',
                       fontproperties=util_fig.label_font)
    axes[1].text(0.05, 0.85, 'Obs', color='r', transform=axes[1].transAxes,
                 fontproperties=util_fig.text_font)
    axes[1].text(0.05, 0.65, 'Model', color='k', transform=axes[1].transAxes,
                 fontproperties=util_fig.text_font)

    axes[2].hist(eps_obs, histtype='step', color='r', lw=lw, bins=bins,
                 range=[0.0, 0.5], normed=True, cumulative=True)
    axes[2].hist(eps, histtype='step', color='k', lw=lw, bins=bins,
                 range=[0.0, 0.5], normed=True, cumulative=True)
    axes[2].set_xlim([0.0, 0.5])
    axes[2].set_ylabel('cumulative',
                       fontproperties=util_fig.label_font)
    axes[2].set_xlabel('$\mathbf{\epsilon}$',
                       fontproperties=util_fig.label_font)
    axes[2].text(0.05, 0.85, 'Obs', color='r', transform=axes[2].transAxes,
                 fontproperties=util_fig.text_font)
    axes[2].text(0.05, 0.65, 'Model', color='k', transform=axes[2].transAxes,
                 fontproperties=util_fig.text_font)

    ax_eps = fig.add_axes([0.28, 0.25, 0.2, 0.3])
    ax_eps.set_xlabel(r'$\mathbf{\varepsilon}$',
                      fontproperties=util_fig.ticks_font1)
    ax_eps.hist(eps_obs, range=xlim, bins=bins, color='r',
                normed=True, lw=3)
    ax_eps.hist(eps, range=xlim, bins=bins, color='k', normed=True,
                histtype='step', lw=3)

    ax_Psai = fig.add_axes([0.28, 0.65, 0.2, 0.3])
    ax_Psai.set_xlabel('$\mathbf{\Delta PA}$',
                       fontproperties=util_fig.ticks_font1)
    ax_Psai.hist(Psai_obs, range=ylim, bins=bins, color='r',
                 normed=True, lw=3)
    ax_Psai.hist(Psai, range=ylim, bins=bins, color='k', normed=True,
                histtype='step', lw=3)

    ax_Psai.set_yticklabels('')
    ax_eps.set_yticklabels('')
    ax_eps.xaxis.set_label_coords(0.5, -0.18)
    util_fig.set_labels(ax_eps, font=util_fig.ticks_font1)
    util_fig.set_labels(ax_Psai, font=util_fig.ticks_font1)

    for ax in axes:
        util_fig.set_labels(ax)
    fig.savefig('{}/{}'.format(outpath, figname), dpi=100)
    return
