#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: franx.py
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
import pickle
import matplotlib.pyplot as plt
import numpy as np
with open('rst.dat') as f:
    rst = pickle.load(f)
fig, axes = plt.subplots(figsize=(6, 3))
axes.plot(np.degrees(rst['Psai']), rst['eps'], '.k', alpha=0.03,
          markersize=1.0)
axes.set_ylim([0.0, 0.6])
axes.set_xlim([0.0, 90.0])
fig.savefig('Frank_A.png')
