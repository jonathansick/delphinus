#!/usr/bin/env python
# encoding: utf-8
"""
Tools for working with artificial stars
"""

import os

import numpy as np
from scipy.spatial import cKDTree as KDTree


class StarList(object):
    """Make and write an artifcial star list.
    
    nImages : int
        Number of images expected.
    """
    def __init__(self, nImages):
        super(StarList, self).__init__()
        self.nImages = nImages
        self._groups = []  # one item for each add_stars() call.
    
    def add_stars(self, x, y, counts, ext=0, chip=1):
        """Add a list of stars for a given `ext` and `chip`.
        
        x,y : ndarray
            Pixel coordinates of artificial stars in frame of the reference
            image (or first image).
        counts : tuple
            Artificial stars brightnesses, in counts, for each image.
            The order must be consistent with the ordering of each image
            photometered in the original DOLPHOT run.
        ext : int
            Extension number, typically 0 for main image.
        chip : int
            Chip number, typically 1.
        """
        # TODO add input checking
        # Stack input
        nStars = len(x)
        stars = np.empty([nStars, 4 + self.nImages], dtype=np.float)
        stars[:, 0] = ext
        stars[:, 1] = chip
        stars[:, 2] = x
        stars[:, 3] = y
        for i, c in enumerate(counts):
            stars[:, i + 4] = c

        self._groups.append(stars)

    def add_stars_mags(self, x, y, mags, zps=25, exptimes=1., ext=0, chip=1):
        """Identical to :func:`add_stars` except that star brightnesses can
        be given as magnitudes. Zeropoints and exposure times will translate
        those magnitudes into counts on each image.
        """
        nImages = len(mags)
        if not isinstance(zps, (tuple, list)):
            zps = [zps] * nImages
        if not isinstance(exptimes, (tuple, list)):
            exptimes = [exptimes] * nImages
        counts = []
        for mag, zp, exptime in zip(mags, zps, exptimes):
            c = 10. ** (-0.4 * (mag - zp)) * exptime
            counts.append(c)
        self.add_stars(x, y, counts, ext=ext, chip=chip)
    
    def write(self, path):
        """Write articial star list to `path`."""
        # Prep output path
        dirname = os.path.dirname(path)
        if (dirname is not "") and (not os.path.exists(dirname)):
            os.makedirs(dirname)
        if os.path.exists(path): os.remove(path)

        # Stack Data
        allStars = np.vstack(tuple(self._groups))

        # Write star list
        fmt = ['%i', '%i', '%.3f', '%.3f']
        for i in xrange(self.nImages):
            fmt.append("%.6f")
        np.savetxt(path, allStars, delimiter=' ', newline='\n', fmt=tuple(fmt))


class ASTReducer(object):
    """Reduce artificial star tests to yield error estimates for individual
    stars, completeness estimates for individual stars, and 50% completeness
    limits for individual bands.

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

    def compute_errors(self, mag_err_lim=None, dx_lim=None):
        """Estimates errors and completeness per star.
        
        Load photometry from fake table (from same chip, ext as primary data.
        For each star in the phot table, get its magnitude.
        Use a kdtree to get the N most similar stars; compute statistics
        """
        mag_errors = self._f.mag_errors()  # diffs nstars x nimages
        recovered = self._f.recovered(mag_err_lim=mag_err_lim, dx_lim=dx_lim)
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

    def completeness_limits(self, frac=0.5, mag_err_lim=None, dx_lim=None):
        """Compute the completeness limit for each image. The magnitude at
        the completeness limit is saved as a an attribute to the phot table
        in the HDF5 file.

        frac : float
            Scalar fractional level of completeness. For example, 0.5 is the
            50% completeness limit.
        mag_err_lim : float
            Maximum absolute difference in magnitudes, in any band, for the
            star to be considered recovered.
        dx_lim : float
            Maximum distance between a fake star's input site and its
            observed site for the fake star to be considered recovered.
        """
        comps = self._f.completeness_limits(mag_err_lim=mag_err_lim,
                dx_lim=dx_lim)
        self._p.set_metadata("completeness", comps)
        return comps
