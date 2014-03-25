#!/usr/bin/env python
# encoding: utf-8
"""
Tools for working with artificial stars
"""

import numpy as np
from scipy.spatial import cKDTree as KDTree


class ASTReducer(object):
    """Reduce artificial star tests to yield error estimates for individual
    stars, completeness estimates for individual stars, and 50% completeness
    limits for individual bands.

    Parameters
    ----------

    fakeReader : :class:`delphinus.phottable.FakeReader`
        A `FakeReader` instance with all artificial stars. Note that multiple
        `FakeReader` instances to can be added together to combine multiple
        artificial star tests.
    photTable : :class:`delphinus.phottable.DolphotTable`
        A `DolphotTable` instance.
    """
    def __init__(self, fakeReader, photTable):
        super(ASTReducer, self).__init__()
        self._f = fakeReader
        self._p = photTable

    def compute_errors(self, mag_err_lim=None, dx_lim=None, qcfunc=None):
        """Estimates errors and completeness per star.
        
        Load photometry from fake table (from same chip, ext as primary data.
        For each star in the phot table, get its magnitude.
        Use a kdtree to get the N most similar stars; compute statistics

        Parameters
        ----------

        frac : float
            Scalar fractional level of completeness. For example, 0.5 is the
            50% completeness limit.
        mag_err_lim : float
            Maximum absolute difference in magnitudes, in any band, for the
            star to be considered recovered.
        dx_lim : float
            Maximum distance between a fake star's input site and its
            observed site for the fake star to be considered recovered.
        qcfunc :
            Callback function for applying quality cuts while assessing
            completeness.
        """
        mag_errors = self._f.mag_errors()  # diffs nstars x nimages
        recovered = self._f.recovered(mag_err_lim=mag_err_lim, dx_lim=dx_lim,
                qcfunc=qcfunc)
        tree = KDTree(self._f.data['mag'])
        obs_mags = np.array([row['mag']
            for row in self._p.photTable.iterrows()])
        dists, indices = tree.query(obs_mags,
                k=100)
                # distance_upper_bound=mag_err_lim)
        nObs = obs_mags.shape[0]
        nImages = obs_mags.shape[1]
        sigmas = np.empty([nObs, nImages])
        comps = np.empty(nObs)
        for i in xrange(nObs):
            if np.any(obs_mags[i] > 50.):
                for j in xrange(nImages):
                    sigmas[i, j] = np.nan
                comps[i] = np.nan
                continue
            idx = indices[i, :].flatten()
            for j in xrange(nImages):
                # Estimate uncertainty in this band (image index)
                sigmas[i, j] = np.std(mag_errors[idx, j])
            # Estimate completeness for this star
            c = recovered[indices[i, :]]
            comps[i] = np.float(c.sum()) / len(c)

        # insert errors into the HDF5 table (need to make a new column
        self._p.add_column("ast_mag_err", sigmas)
        # insert completeness for this star
        self._p.add_column("comp", comps)

    def completeness_limits(self, frac=0.5, mag_err_lim=None, dx_lim=None,
            qcfunc=None):
        """Compute the completeness limit for each image. The magnitude at
        the completeness limit is saved as a an attribute to the phot table
        in the HDF5 file.

        Parameters
        ----------

        frac : float
            Scalar fractional level of completeness. For example, 0.5 is the
            50% completeness limit.
        mag_err_lim : float
            Maximum absolute difference in magnitudes, in any band, for the
            star to be considered recovered.
        dx_lim : float
            Maximum distance between a fake star's input site and its
            observed site for the fake star to be considered recovered.
        qcfunc :
            Callback function for applying quality cuts while assessing
            completeness.
        """
        comps = self._f.completeness_limits(mag_err_lim=mag_err_lim,
                dx_lim=dx_lim, qcfunc=qcfunc)
        self._p.set_metadata("completeness", comps)
        return comps
