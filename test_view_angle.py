#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test_view_angle.py
# Author: Hongyu Li <lhy88562189@gmail.com>
# Date: 31.07.2017
# Last Modified: 31.07.2017
# ============================================================================
#  DESCRIPTION: test random view angle generator
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
# ORGANIZATION:
#      VERSION: 0.0
# ============================================================================
import numpy as np
import util_angle
import matplotlib.pyplot as plt
theta, phi = util_angle.get_view_angle(100000)
fig, axes = plt.subplots(3, 1, figsize=(6, 6))
axes[0].hist(theta, range=[0.0, np.pi], bins=100, histtype='step')
axes[1].hist(np.cos(theta), range=[0.0, 1.0], bins=100, histtype='step')
axes[2].hist(phi, range=[0.0, 2.0*np.pi], bins=100, histtype='step')
plt.show()
