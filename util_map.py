#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: util_map.py
# Author: Hongyu Li <lhy88562189@gmail.com>
# Date: 28.07.2017
# Last Modified: 28.07.2017
# ============================================================================
#  DESCRIPTION: create a basemap using Orthographic projection, and map (x, y)
#               to (theta, phi)
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
# ORGANIZATION:
#      VERSION: 0.0
# ============================================================================
from mpl_toolkits.basemap import Basemap
import numpy as np
from JAM.utils import util_fig
import matplotlib.pyplot as plt


def convert_lonlat_phitheta(lon, lat):
    phi = np.zeros_like(lon)
    theta = np.zeros_like(lat)
    i_lon_east = lon > 0.0
    i_lon_west = lon < 0.0
    # i_lat_north = lat > 0.0
    # i_lat_south = lat < 0.0
    phi[i_lon_east] = np.radians(lon[i_lon_east])
    phi[i_lon_west] = np.radians(lon[i_lon_west]) + 2.0*np.pi
    theta = np.pi/2.0 - np.radians(lat)
    return phi, theta


def make_base(lon_0=45.0, lat_0=30.0, ngrid=100):
    m = Basemap(projection='ortho', lon_0=lon_0, lat_0=lat_0, resolution='l')
    # ----------------------------- plot ------------------------------------
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.92,
                        top=0.92, wspace=0.0, hspace=0.0)
    # draw parallels and meridians.
    m.drawparallels(np.arange(-80., 80., 20.))
    m.drawmeridians(np.arange(-180., 180., 20.))
    # m.drawmeridians(np.array([0.0, -10.]))
    # m.drawparallels(np.array([0.0, -10.]))

    X_x, X_y = m(0.0, 0.0)  # cross point of X axis
    Y_x, Y_y = m(90.0, 0.0)  # cross point of Y axis
    Z_x, Z_y = m(0.0, 90.0)  # cross point of Z axis
    ctr_x, ctr_y = m(lon_0, lat_0)  # x, y coordinates of the origin
    # axis vectors
    x_axis_x = np.array([ctr_x, X_x]) + np.array([0.0, X_x-ctr_x]) * 0.4
    x_axis_y = np.array([ctr_y, X_y]) + np.array([0.0, X_y-ctr_y]) * 0.4
    y_axis_x = np.array([ctr_x, Y_x]) + np.array([0.0, Y_x-ctr_x]) * 0.4
    y_axis_y = np.array([ctr_y, Y_y]) + np.array([0.0, Y_y-ctr_y]) * 0.4
    z_axis_x = np.array([ctr_x, Z_x]) + np.array([0.0, Z_x-ctr_x]) * 0.4
    z_axis_y = np.array([ctr_y, Z_y]) + np.array([0.0, Z_y-ctr_y]) * 0.4
    # plot axes and their cross point
    axes.plot(x_axis_x, x_axis_y, 'k--', lw=2)
    axes.plot(y_axis_x, y_axis_y, 'k--', lw=2)
    axes.plot(z_axis_x, z_axis_y, 'k--', lw=2)
    axes.plot(X_x, X_y, 'xr')
    axes.plot(Y_x, Y_y, 'xr')
    axes.plot(Z_x, Z_y, 'xr')
    axes.text(0.00, 0.2, 'X', transform=axes.transAxes,
              fontproperties=util_fig.text_font)
    axes.text(0.99, 0.2, 'Y', transform=axes.transAxes,
              fontproperties=util_fig.text_font)
    axes.text(0.49, 1.02, 'Z', transform=axes.transAxes,
              fontproperties=util_fig.text_font)
    # map x, y to theta and phi
    x = np.linspace(0.0, m.xmax, ngrid)
    y = np.linspace(0.0, m.ymax, ngrid)
    X, Y = np.meshgrid(x, y)
    lon, lat = m(X, Y, inverse=True)
    mask = (lon > 1e10) * (lat > 1e10)
    lon[mask] = np.nan
    lat[mask] = np.nan
    # X[mask] = np.nan
    # Y[mask] = np.nan
    # axes.plot(X, Y, 'r.')
    phi, theta = convert_lonlat_phitheta(lon, lat)
    phi[mask] = np.nan
    theta[mask] = np.nan
    return m, fig, axes, theta, phi, mask, X, Y
