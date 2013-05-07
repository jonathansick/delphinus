#!/usr/bin/env python
# encoding: utf-8
"""
Class for working with Dolphot output data; particularly converting text
output into HDF5 files and providing nice hooks into those HDF5 tables.
"""

import os
import numpy as np
import numpy.lib.recfunctions as recf
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


class BasePhotReader(object):
    """Base class for reading Dolphot photometry output files."""
    N_IMAGE_COLS = 17
    N_GLOBAL_COLS = 11
    GLOBAL_COL_OFFSET = 0

    def __init__(self, filepath, nImages, refImagePath=None):
        super(BasePhotReader, self).__init__()
        self.filepath = filepath
        self.nImages = nImages
        self.referenceImagePath = refImagePath
        self._read()

    def extract_additional_columns(self, data, nImages, nStars):
        """User hook for extracting additional columns from the photometry
        output file.
        """
        pass
    
    def _read(self):
        """Pipeline for reading DOLPHOT photometry output."""
        data = np.loadtxt(self.filepath)
        nStars = data.shape[0]
        self._extract_global_cols(data, self.GLOBAL_COL_OFFSET, nStars)
        self._extract_image_phot_cols(data,
                self.GLOBAL_COL_OFFSET + self.N_GLOBAL_COLS,
                self.nImages, nStars)
        self.extract_additional_columns(data, self.nImages, nStars)  # hook
        self.combine_structured_array()

    def _extract_image_phot_cols(self, data, offset, nImages, nStars):
        """Extract output for image-specific photometry columns."""
        dt = [('counts', np.float, self.nImages),
                ('sky', np.float, self.nImages),
                ('norm_count_rate', np.float, self.nImages),
                ('norm_count_rate_err', np.float, self.nImages),
                ('mag', np.float, self.nImages),
                ('mag_err', np.float, self.nImages),
                ('chi', np.float, self.nImages),
                ('sn', np.float, self.nImages),
                ('sharp', np.float, self.nImages),
                ('round', np.float, self.nImages),
                ('crowding', np.float, self.nImages),
                ('fwhm', np.float, self.nImages),
                ('ecc', np.float, self.nImages),
                ('psf_a', np.float, self.nImages),
                ('psf_b', np.float, self.nImages),
                ('psf_c', np.float, self.nImages),
                ('quality', np.float, self.nImages)]
        self._idata = np.empty(nStars, dtype=np.dtype(dt))
        for i in range(self.nImages):
            for j in range(self.N_IMAGE_COLS):
                k = offset + i * self.N_IMAGE_COLS + j
                colname = dt[j][0]
                self._idata[colname][:, i] = data[:, k]

    def _extract_global_cols(self, data, offset, nStars):
        """Extract output for global image columns."""
        dt = [('ext', np.int),
                ('chip', np.int),
                ('x', np.int),
                ('y', np.int),
                ('ra', np.float),
                ('dec', np.float),
                ('ref_chi', np.float),
                ('ref_sn', np.float),
                ('ref_sharp', np.float),
                ('ref_round', np.float),
                ('major_ax', np.int),
                ('ref_crowding', np.float),
                ('type', np.int)]
        self._gdata = np.empty(nStars, dtype=np.dtype(dt))
        for i in range(self.N_GLOBAL_COLS):
            j = i + offset
            colname = dt[i][0]
            self._gdata[colname] = data[:, j]

    def _fill_radec(self):
        """Compute RA/Dec columns, if a reference image is specified."""
        if self.referenceImagePath is not None:
            refHead = getheader(self.referenceImagePath)
            wcs = WCS(refHead)
            ra, dec = wcs.all_pix2sky(self._gdata['x'], self._gdata['y'], 1)
            self._gdata['ra'] = ra
            self._gdata['dec'] = dec
        else:
            self._gdata['ra'][:] = np.nan
            self._gdata['dec'][:] = np.nan

    def combine_structured_array(self):
        """docstring for combine_structured_array"""
        arrays = [self._gdata, self._idata]
        self.data = recf.merge_arrays(arrays, flatten=True, usemask=False)


class FakeReader(BasePhotReader):
    """Read Dolphot's .fake artificial star output."""
    N_FAKE_GLOBAL_COLS = 4
    N_FAKE_IMAGE_COLS = 2

    def __init__(self, filepath, nImages, refImagePath=None):
        self.GLOBAL_COL_OFFSET = self.N_FAKE_GLOBAL_COLS \
                + nImages * self.N_FAKE_IMAGE_COLS
        super(FakeReader, self).__init__(filepath, nImages,
                refImagePath=refImagePath)

    def __add__(self, other):
        """Return a concatenated FakeReader (concatenates data)."""
        self.data = np.concatenate((self.data, other.data))
        return self
    
    def extract_additional_columns(self, data, nImages, nStars):
        """Reads additional columns for .fake output."""
        self._extract_fake_global_cols(data, nStars)
        self._extract_fake_phot_cols(data, nImages, nStars)

    def _extract_fake_global_cols(self, data, nStars):
        """Extract global columns at beginning of .fake files."""
        dt = [('fake_ext', np.int),
                ('fake_chip', np.int),
                ('fake_x', np.float),
                ('fake_y', np.float),
                ('fake_ra', np.float),
                ('fake_dec', np.float)]
        self._fgdata = np.empty(nStars, dtype=np.dtype(dt))
        for i in range(self.N_FAKE_GLOBAL_COLS):
            colname = dt[i][0]
            self._fgdata[colname] = data[:, i]
        self._fake_fill_radec()

    def _extract_fake_phot_cols(self, data, nImages, nStars):
        """Extract input photometry columns at beginning of .fake files
        for each image.
        """
        dt = [('fake_count', np.float, nImages),
                ('fake_mag', np.float, nImages)]
        self._fidata = np.empty(nStars, dtype=np.dtype(dt))
        for i in range(self.nImages):
            for j in range(self.N_FAKE_IMAGE_COLS):
                k = self.N_FAKE_GLOBAL_COLS + i * self.N_FAKE_IMAGE_COLS + j
                colname = dt[j][0]
                self._fidata[colname][:, i] = data[:, k]

    def _fake_fill_radec(self):
        """Compute RA/Dec columns, if a reference image is specified."""
        # TODO refactor this code against _fill_radec()
        if self.referenceImagePath is not None:
            refHead = getheader(self.referenceImagePath)
            wcs = WCS(refHead)
            ra, dec = wcs.all_pix2sky(self._gdata['x'], self._gdata['y'], 1)
            self._fgdata['fake_ra'] = ra
            self._fgdata['fake_dec'] = dec
        else:
            self._fgdata['fake_ra'][:] = np.nan
            self._fgdata['fake_dec'][:] = np.nan

    def combine_structured_array(self):
        """docstring for combine_structured_array"""
        arrays = [self._fgdata, self._fidata, self._gdata, self._idata]
        self.data = recf.merge_arrays(arrays, flatten=True, usemask=False)

    def mag_errors(self):
        """Compute output-input magnitude difference for AST.
        """
        imageResults = []
        for n in xrange(self.nImages):
            fakeMag = self.data['fake_mag'][:, n]
            obsMag = self.data['mag'][:, n]
            imageResults.append((fakeMag, obsMag - fakeMag))
        return imageResults

    def position_errors(self, magIndex=0):
        """Prototype for computing position errors for AST as the
        Euclidean distance between input and output (x,y) coordinates.
        """
        fakeMag = self.data['fake_mag'][:, magIndex]
        inputX = self.data['fake_x']
        inputY = self.data['fake_y']
        obsX = self.data['x']
        obsY = self.data['y']
        dx = np.hypot(inputX - obsX, inputY - obsY)
        return fakeMag, dx

    def completeness(self, dmag=0.2, magErrLim=None, dxLim=None):
        """Prototype for reporting completeness in each image, as a function
        of input magnitude using DOLPHOT's metric for star recovery success.
        """
        imageResults = []
        if dxLim is not None:
            k, dx = self.position_errors()
        for n in xrange(self.nImages):
            fakeMag = self.data['fake_mag'][:, n]
            obsMag = self.data['mag'][:, n]
            # Dolphot gives unrecovered stars a magnitude of 99. This should
            # safely distinguish those stars.
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

    def metrics(self, magRange, magErrLim=None, dxLim=None):
        """Makes scalar metrics of artificial stars in an image.
        
        For each image, results a tuple (RMS mag error, completeness fraction).
        """
        imageResults = []
        if dxLim is not None:
            k, dx = self.position_errors()
        for n in xrange(self.nImages):
            fakeMag = self.data['fake_mag'][:, n]
            obsMag = self.data['mag'][:, n]
            err = np.abs(fakeMag - obsMag)
            # Dolphot gives unrecovered stars a magnitude of 99. This should
            # safely distinguish those stars.
            recovered = obsMag < 50.
            if magErrLim is not None:
                recovered = recovered & (err < magErrLim)
            if dxLim is not None:
                recovered = recovered & (dx < dxLim)
            recovered = np.array(recovered, dtype=np.float)
            # Find stars in magnitude range
            minMask = fakeMag > min(magRange)
            maxMask = fakeMag < max(magRange)
            found = obsMag < 50.
            inds = np.where(minMask & maxMask)[0]
            indsMeasured = np.where(minMask & maxMask & found)[0]
            comp = float(np.sum(recovered[inds]) / float(len(inds)))
            rms = float(np.std(err[indsMeasured]))
            imageResults.append((rms, comp))
        return imageResults


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
        # Read phot file
        nImages = len(images)
        if referenceImage is not None:
            refPath = referenceImage['path']
        else:
            # use the first image as a reference instead
            refPath = images[0]['path']
        reader = BasePhotReader(photPath, nImages, refImagePath=refPath)

        # Insert the structured numpy array into a new HDF5 table
        title = os.path.splitext(os.path.basename(tablePath))[0]
        if os.path.exists(tablePath): os.remove(tablePath)
        h5 = tables.openFile(tablePath, mode="w", title=title)
        photTable = h5.createTable("/", 'phot', reader.data.dtype,
                "Photometry Catalog")
        photTable.append(reader.data)
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


if __name__ == '__main__':
    photPath = "/Users/jsick/Dropbox/_dolphot/517eef6ce8f07284365c6156"
    photTable = BasePhotReader(photPath, 2, refImagePath=None) 
    print photTable.data.dtype
    print photTable.data['chip']
    fakePath = "/Users/jsick/Dropbox/_dolphot/517eef6ce8f07284365c6156.fake"
    fakeTable = FakeReader(fakePath, 2, refImagePath=None)
    print fakeTable.data['fake_mag'][0]
    print fakeTable.data['fake_count'][0]
    print fakeTable.data['mag'][0]
    print fakeTable.data['counts'][0]
    print fakeTable.data['fake_x'][0]
    print fakeTable.data['fake_y'][0]
    print fakeTable.data['x'][0]
    print fakeTable.data['y'][0]
    fakeTable2 = FakeReader(fakePath, 2, refImagePath=None)
    print fakeTable.data.dtype
    print fakeTable.mag_errors()
    print fakeTable.position_errors()
    print fakeTable.completeness()
    print len(fakeTable.data)
    concatTable = fakeTable + fakeTable2
    print len(concatTable.data)
    print fakeTable.metrics([17., 18.], magErrLim=0.2)
    print fakeTable.metrics([19., 19.5], magErrLim=0.2)
