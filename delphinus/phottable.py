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

import lfdiagnostics


class DolphotTable(object):
    """Represents the output from Dolphot in an HDF5 table."""
    def __init__(self, hdfPath):
        super(DolphotTable, self).__init__()
        self.hdfPath = hdfPath
        self.hdf = tables.openFile(self.hdfPath)
        self.photTable = self.hdf.root.phot
    
    @classmethod
    def make(cls, tablePath, images, referenceImage, photPath, psfsPath,
            apcorPath, execTime=None):
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
        photTable.attrs.image_bands = [im['band'] for im in images]
        if execTime is not None:
            photTable.attrs.exec_time = execTime
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
        """List of image keys in photometry, ordered as in catalog.
        
        Image keys are strings used to represent an image in your pipeline.
        """
        return self.photTable.attrs.image_keys

    @property
    def image_bands(self):
        """List of image bandpasses, ordered with :meth:`self.image_paths`."""
        return self.photTable.attrs.image_bands

    def plot_luminosity_function_diagnostics(self, plotPathRoot,
            magLim=None, fmt="pdf"):
        """docstring for plot_luminosity_function_diagnostics"""
        plotDir = os.path.dirname(plotPathRoot)
        if plotDir is not "" and not os.path.exists(plotDir):
            os.makedirs(plotDir)

        for i, (imageKey, band) in enumerate(zip(self.image_keys,
                self.image_bands)):
            plotPath = "%s_%s_%s.%s" % (plotPathRoot, imageKey, band, fmt)
            lfdiagnostics.make_diagnostic_plot(self, i, imageKey,
                    band, fmt, plotPath, magLim)


class WIRCamFakeTable(object):
    """Quick and dirty interface to DOLPHOT's .fake output files from
    artificial star tests.
    
    .. todo:: Enable this AST photometry to be embedded in a dolphot HDF5
       output table.
    """
    def __init__(self, fakePath, refImagePath=None):
        super(WIRCamFakeTable, self).__init__()
        self.path = fakePath
        self._data = self._read(refImagePath)

    def _read(self, refImagePath=None):
        """Read .fake file."""
        n = 2  # two images in WIRCam testing
        # TODO add columns for fake stars/recovery
        colDefs = np.dtype([
            # Fake star columns
            ('fake_ext', np.int), ('fake_chip', np.int),
            ('fake_x', np.float), ('fake_y', np.float),
            ('fake_count_1', np.float), ('fake_mag_1', np.float),
            ('fake_count_2', np.float), ('fake_mag_2', np.float),
            # Regular columns
            ('ext', np.int), ('chip', np.int),
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
        # Prepended columns for fake stars
        dataTypes = [np.int, np.int, np.float, np.float, # fake
                np.float, np.float, np.float, np.float,  # fake counts/mags
                np.int, np.int, np.int, np.int,
                np.float, np.float, np.float, np.float, np.int,
                np.float, np.int]
        colNames = ['fake_ext', 'fake_chip', 'fake_x', 'fake_y',
                'fake_count_1', 'fake_mag_1', 'fake_count_2', 'fake_mag_2',
                'ext', 'chip', 'x', 'y', 'ref_chi', 'ref_sn',
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
        data = np.loadtxt(self.path)
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
        if refImagePath is not None:
            refHead = getheader(refImagePath)
            wcs = WCS(refHead)
            ra, dec = wcs.all_pix2sky(dataRecArray['x'], dataRecArray['y'], 1)
            dataRecArray['ra'] = ra
            dataRecArray['dec'] = dec

        return dataRecArray

    def mag_errors(self):
        """Prototype for computing output-input magnitudes for two-image AST.
        """
        imageResults = []
        for n in xrange(2):
            fakeMag = self._data['fake_mag_%i' % (n + 1, )]
            obsMag = self._data['mag'][:, n]
            imageResults.append((fakeMag, obsMag - fakeMag))
        return imageResults

    def position_errors(self):
        """Prototype for computing position errors for two-image AST as the
        Euclidean distance between input and output (x,y) coordinates.
        """
        fakeMagK = self._data['fake_mag_2']
        inputX = self._data['fake_x']
        inputY = self._data['fake_y']
        obsX = self._data['x']
        obsY = self._data['y']
        dx = np.hypot(inputX - obsX, inputY - obsY)
        return fakeMagK, dx

    def completeness(self, dmag=0.2, magErrLim=None, dxLim=None):
        """Prototype for reporting completeness in each image, as a function
        of input magnitude using DOLPHOT's metric for star recovery success.
        """
        imageResults = []
        if dxLim is not None:
            k, dx = self.position_errors()
        for n in xrange(2):
            fakeMag = self._data['fake_mag_%i' % (n + 1, )]
            obsMag = self._data['mag'][:, n]
            # Dolphot gives unrecovered stars a magnitude of 99. This should
            # safely distinguish those stars.
            # recovered = np.array(obsMag < 50., dtype=np.float)
            recovered = obsMag < 50.
            if magErrLim is not None:
                err = np.abs(fakeMag - obsMag)
                recovered = recovered & (err < magErrLim)
            if dxLim is not None:
                recovered = recovered & (dx < dxLim)
            recovered = np.array(recovered, dtype=np.float)
            bins = np.arange(fakeMag.min(), fakeMag.max(), dmag)
            inds = np.digitize(fakeMag, bins)
            rec = np.bincount(inds, weights=recovered, minlength=None)
            tot = np.bincount(inds, weights=None, minlength=None)
            comp = rec / tot
            # FIXME need to resolve issue with histogram edges
            imageResults.append((bins, comp[1:]))
        return imageResults


if __name__ == '__main__':
    fakePath = "/Users/jsick/Dropbox/_dolphot/517eef6ce8f07284365c6156.fake"
    fakeTable = WIRCamFakeTable(fakePath)
    print fakeTable.mag_errors()
    print fakeTable.position_errors()
    print fakeTable.completeness()
