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


Psai_int = np.radians(72.0)
m, fig, axes, theta, phi, mask, X, Y = \
    util_map.make_base(lon_0=45.0, lat_0=30.0, ngrid=1001)
Gamma_kin = util_angle.get_Gamma_kin(theta, phi, Psai_int)
levels = np.arange(15.0, 180.0, 15.0)
Gamma_kin_deg = np.degrees(Gamma_kin)
i_mask = (Gamma_kin_deg > 175.0) + (Gamma_kin_deg < 5.0)
Gamma_kin_deg[i_mask] = np.nan
CS = axes.contour(X, Y, Gamma_kin_deg, levels=levels, linewidths=0.5,
                  colors='b')
CS.clabel(inline=1, fmt='%.0f', color='r')
# axes.plot(X, Y, '.')
fig.savefig('map.pdf')
# plt.clf()
# ax = plt.gca()
# ax.imshow(Gamma_kin_deg, origin='bottom')
# fig.savefig('color.pdf')
