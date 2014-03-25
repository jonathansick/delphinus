#!/usr/bin/env python
# encoding: utf-8
"""
Tools for working with artificial star test results.

This module exists for computations that need to work with both PhotTable
and FakeTable data sets, without modifying either. These funtions can be
called, for example from a PhotTable where it would be expected that the
result is cached in the table.
"""

import numpy as np
from scipy.spatial import cKDTree as KDTree


def estimate_errors(phot_tbl, fake_tbl,
        mag_err_lim=None, dx_lim=None, qcfunc=None):
    """Estimates errors and completeness per observed star given the paired
    artificial star test data.
    
    Load photometry from fake table (from same chip, ext as primary data.
    For each star in the phot table, get its magnitude.
    Use a kdtree to get the N most similar stars; compute statistics

    Parameters
    ----------
    phot_tbl : :class:`delphinus.table.PhotTable`
        Photometry table.
    fake_tbl : :class:`delphinus.table.FakeTable`
        Artificial star test table.
    mag_err_lim : float
        Maximum absolute difference in magnitudes, in any band, for the
        star to be considered recovered.
    dx_lim : float
        Maximum distance between a fake star's input site and its
        observed site for the fake star to be considered recovered.
    qcfunc :
        Callback function for applying quality cuts while assessing
        completeness.

    Returns
    -------
    comps : ndarray
        Array of completion rate estimates for each star in `phot_tbl`.
    sigmas : ndarray
        Array of uncertainties for observed magnitudes.
        Shape `(n_obs, n_images)`.
    """
    mag_errors = np.column_stack([fake_tbl.mag_errors(n)
        for n in range(fake_tbl.n_images)])
    fake_input_mags = np.column_stack([fake_tbl.image_col('fake_mag', n)
        for n in range(fake_tbl.n_images)])
    obs_mags = np.column_stack([phot_tbl.image_col('mag', n)
        for n in range(phot_tbl.n_images)])
    recovered = fake_tbl.recovered(mag_err_lim=mag_err_lim, dx_lim=dx_lim,
            qcfunc=qcfunc)

    tree = KDTree(fake_input_mags)
    dists, indices = tree.query(obs_mags, k=100)
    n_obs = obs_mags.shape[0]
    n_images = obs_mags.shape[1]
    sigmas = np.empty([n_obs, n_images])
    comps = np.empty(n_obs)
    for i in xrange(n_obs):
        if np.any(obs_mags[i] > 50.):
            for j in xrange(n_images):
                sigmas[i, j] = np.nan
            comps[i] = np.nan
            continue
        idx = indices[i, :].flatten()
        for j in xrange(n_images):
            # Estimate uncertainty in this band (image index)
            sigmas[i, j] = np.std(mag_errors[idx, j])
        # Estimate completeness for this star
        c = recovered[indices[i, :]]
        comps[i] = np.float(c.sum()) / len(c)

    return comps, sigmas
