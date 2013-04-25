#!/usr/bin/env python
# encoding: utf-8
"""
Tools for working with artificial stars
"""

import os

import numpy as np


class StarList(object):
    """Make and write an artifcial star list.
    
    nImages : int
        Number of images expected.
    """
    def __init__(self, nImages):
        super(StarList, self).__init__()
        self.nImages = nImages
        self._groups = []  # one item for each add_stars() call.
    
    def add_stars(self, x, y, ext=0, chip=1, *counts):
        """Add a list of stars for a given `ext` and `chip`.
        
        x,y : ndarray
            Pixel coordinates of artificial stars in frame of the reference
            image (or first image).
        *counts : ndarray
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
    
    def write(self, path):
        """Write articial star list to `path`."""
        # Prep output path
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname): os.makedirs(dirname)
        if os.path.exists(path): os.remove(path)

        # Stack Data
        allStars = np.vstack(tuple(self._groups))

        # Write star list
        fmt = ['%i', '%i', '%.3f' '%.3f']
        for i in xrange(self.nImages):
            fmt.append("%.6f")
        np.savetxt(path, allStars, delimeter=' ', newline='\n', fmt=tuple(fmt))
