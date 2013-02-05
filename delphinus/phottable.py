#!/usr/bin/env python
# encoding: utf-8
"""
Class for working with Dolphot output data; particularly converting text
output into HDF5 files and providing nice hooks into those HDF5 tables.
"""

import os
import numpy as np
try:
    from astropy.wcs import WCS
except ImportError:
    from pywcs import WCS
try:
    from astropy.io.fits import getheader
except ImportError:
    from pyfits import getheader
import tables


class DolphotTable(object):
    """Represents the output from Dolphot in an HDF5 table."""
    def __init__(self, hdfPath):
        super(DolphotTable, self).__init__()
        self.hdfPath = hdfPath
        self.hdf = tables.openFile(self.hdfPath)
        self.photTable = self.hdf.root.phot
    
    @classmethod
    def make(cls, tablePath, images, referenceImage, photPath, psfsPath,
            apcorPath):
        """Initialize a DolphotTable using data from the Dolphot class."""
        n = len(images)
        # Column definitions for the HDF5 table
        # measurements specific to an image are multiplexed in a single
        # column (ie,  a third dimension in the table)
        colDefs = np.dtype([('ext', np.int), ('chip', np.int),
            ('x', np.int), ('y', np.int),
            ('ra', np.float), ('dec', np.float), ('ref_chi', np.float),
            ('ref_sn', np.float), ('ref_sharp', np.float),
            ('ref_round', np.float), ('major_ax', np.int),
            ('ref_crowding', np.float), ('type', np.int),
            ('counts', np.float, n),
            ('sky', np.float, n), ('norm_count_rate', np.float, n),
            ('norm_count_rate_err', np.float, n), ('mag', np.float, n),
            ('mag_err', np.float, n), ('chi', np.float, n),
            ('sn', np.float, n),
            ('sharp', np.float, n), ('round', np.float, n),
            ('crowding', np.float, n), ('fwhm', np.float, n),
            ('ecc', np.float, n), ('psf_a', np.float, n),
            ('psf_b', np.float, n),
            ('psf_c', np.float, n), ('quality', np.int, n)])
        # Data type for numpy load txt
        dataTypes = [np.int, np.int, np.int, np.int,
                np.float, np.float, np.float, np.float, np.int,
                np.float, np.int]
        colNames = ['ext', 'chip', 'x', 'y', 'ref_chi', 'ref_sn',
                'ref_sharp', 'ref_round', 'major_ax', 'ref_crowding', 'type']
        imgDataTypes = [np.float, np.float, np.float,
                np.float, np.float, np.float, np.float, np.float,
                np.float, np.float,
                np.float, np.float,
                np.float, np.float, np.float,
                np.float, np.int]
        imgColNames = ['counts', 'sky', 'norm_count_rate',
                'norm_count_rate_err',
                'mag', 'mag_err', 'chi', 'sn', 'sharp', 'round', 'crowding',
                'fwhm', 'ecc', 'psf_a', 'psf_b', 'psf_c', 'quality']
        for i in xrange(n):
            dataTypes += imgDataTypes
        data = np.loadtxt(photPath)
        nStars = data.shape[0]
        dataRecArray = np.empty(nStars, dtype=colDefs)
        # Get columns for general star data
        for i, colName in enumerate(colNames):
            dataRecArray[colName] = data[:, i]
        # Get columns for measurements from specific images
        for j in xrange(n):
            for i, colName in enumerate(imgColNames):
                k = j * len(imgDataTypes) + len(colNames) + i
                dataRecArray[colName][:, j] = data[:, k]

        # Compute RA and Dec from x,y coords against ref or first image
        if referenceImage is not None:
            refHead = getheader(referenceImage['path'])
        else:
            # use the first image as a reference instead
            refHead = getheader(images[0]['path'])
        wcs = WCS(refHead)
        ra, dec = wcs.all_pix2sky(dataRecArray['x'], dataRecArray['y'], 1)
        dataRecArray['ra'] = ra
        dataRecArray['dec'] = dec

        # Insert the structured numpy array into a new HDF5 table
        title = os.path.splitext(os.path.basename(tablePath))[0]
        if os.path.exists(tablePath): os.remove(tablePath)
        h5 = tables.openFile(tablePath, mode="w", title=title)
        photTable = h5.createTable("/", 'phot', colDefs,
                "Photometry Catalog")
        print dataRecArray.dtype
        photTable.append(dataRecArray)
        # Set meta data for the photometry table
        photTable.attrs.image_paths = [im['path'] for im in images]
        photTable.attrs.image_keys = [im['image_key'] for im in images]
        # Save and close
        photTable.flush()
        h5.close()

        instance = cls(tablePath)
        return instance

    @property
    def image_paths(self):
        """List of image paths in photometry, ordered as in catalog."""
        return self.photTable.attrs.image_paths

    @property
    def image_keys(self):
        """List of image paths in photometry, ordered as in catalog."""
        return self.photTable.attrs.image_keys
