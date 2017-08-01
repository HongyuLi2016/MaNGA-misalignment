#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test_Phi-kin.py
# Author: Hongyu Li <lhy88562189@gmail.com>
# Date: 27.07.2017
# Last Modified: 27.07.2017
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
import util_map
import util_angle
# import matplotlib.pyplot as plt

zeta = 0.6
ksai = 0.5
m, fig, axes, theta, phi, mask, X, Y = \
    util_map.make_base(lon_0=45.0, lat_0=20.0, ngrid=1001)
Gamma_min, eps = util_angle.get_Gamma_min(theta, phi, zeta, ksai)
levels = np.arange(ksai+0.025, 0.98, 0.05)
# i_mask = (Gamma_kin_deg > 175.0) + (Gamma_kin_deg < 5.0)
# Gamma_kin_deg[i_mask] = np.nan
CS = axes.contour(X, Y, eps, levels=levels, linewidths=0.5,
                  colors='b')
CS.clabel(inline=1, fmt='%.3f', color='r')
# axes.plot(X, Y, '.')
fig.savefig('map.pdf')
# plt.clf()
# ax = plt.gca()
# ax.imshow(Gamma_kin_deg)
# fig.savefig('color.pdf')
