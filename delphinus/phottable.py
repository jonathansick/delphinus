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
                # FIXME must be a better way to cast this!
                # temp fix for nimages=1 shape
                if nImages > 1:
                    self._idata[colname][:, i] = data[:, k]
                else:
                    self._idata[colname] = data[:, k]

    def _extract_global_cols(self, data, offset, nStars):
        """Extract output for global image columns."""
        dt = [('ext', np.int),
                ('chip', np.int),
                ('x', np.float),
                ('y', np.float),
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
                # FIXME fix for nimages=1 shape
                if nImages > 1:
                    self._fidata[colname][:, i] = data[:, k]
                else:
                    self._fidata[colname] = data[:, k]
                # self._fidata[colname][:, i] = data[:, k]

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

        Returns
        -------
        An nImage by nStars array of output-input differences.
        """
        diffs = self.data['mag'] - self.data['fake_mag']
        return diffs

    def position_errors(self, magIndex=0):
        """Prototype for computing position errors for AST as the
        Euclidean distance between input and output (x,y) coordinates.

        .. todo:: Should not return fakeMag

        .. todo:: Should cache results
        """
        if self.nImages > 1:
            fakeMag = self.data['fake_mag'][:, magIndex]
        else:
            fakeMag = self.data['fake_mag']
        inputX = self.data['fake_x']
        inputY = self.data['fake_y']
        obsX = self.data['x']
        obsY = self.data['y']
        dx = np.hypot(inputX - obsX, inputY - obsY)
        return fakeMag, dx

    def completeness(self, n, mag_err_lim=None, dx_lim=None,
            frac=0.5, dmag=0.1):
        """Returns magnitude vs completeness fraction for the given image.

        n : int
            Index of image to compute completeness limit for.
        mag_err_lim : float
            Maximum absolute difference in magnitudes, in any band, for the
            star to be considered recovered.
        dx_lim : float
            Maximum distance between a fake star's input site and its
            observed site for the fake star to be considered recovered.
        frac : float
            Scalar fractional level of completeness. For example, 0.5 is the
            50% completeness limit.
        dmag : float
            Bin width (magnitudes) in histogram when establishing completeness
            per bin.
        """
        recovered = self.recovered_in_image(n, mag_err_lim=mag_err_lim,
                dx_lim=dx_lim)
        if self.nImages > 1:
            fakeMag = self.data['fake_mag'][:, n]
        else:
            fakeMag = self.data['fake_mag']
        bins = np.arange(fakeMag.min(), fakeMag.max(), dmag)
        inds = np.digitize(fakeMag, bins)
        rec = np.bincount(inds, weights=recovered, minlength=None)
        tot = np.bincount(inds, weights=None, minlength=None)
        comp = rec / tot
        return bins, comp[1:]

    def completeness_limits(self, mag_err_lim=None, dx_lim=None,
            frac=0.5, dmag=0.1):
        """Compute the completeness limit against each image.
        Returns a list of completeness limits corresponding to each image.
        
        mag_err_lim : float
            Maximum absolute difference in magnitudes, in any band, for the
            star to be considered recovered.
        dx_lim : float
            Maximum distance between a fake star's input site and its
            observed site for the fake star to be considered recovered.
        frac : float
            Scalar fractional level of completeness. For example, 0.5 is the
            50% completeness limit.
        dmag : float
            Bin width (magnitudes) in histogram when establishing completeness
            per bin.
        """
        comps = []
        for n in xrange(self.nImages):
            c = self.completeness_limit_for_image(n,
                    mag_err_lim=mag_err_lim, dx_lim=dx_lim,
                    frac=frac, dmag=dmag)
            comps.append(c)
        return comps

    def completeness_limit_for_image(self, n, mag_err_lim=None, dx_lim=None,
            frac=0.5, dmag=0.1):
        """Compute the completeness limit against each a single image.
        
        n : int
            Index of image to compute completeness limit for.
        mag_err_lim : float
            Maximum absolute difference in magnitudes, in any band, for the
            star to be considered recovered.
        dx_lim : float
            Maximum distance between a fake star's input site and its
            observed site for the fake star to be considered recovered.
        frac : float
            Scalar fractional level of completeness. For example, 0.5 is the
            50% completeness limit.
        dmag : float
            Bin width (magnitudes) in histogram when establishing completeness
            per bin.
        """
        bins, comp = self.completeness(n,
                mag_err_lim=mag_err_lim, dx_lim=dx_lim, dmag=dmag)
        # Solve where completeness crosses the threshold
        # TODO this could be made a lot smarter (i.e., ensure completeess
        # is declining; interpolate between bins; smooth)
        msk = np.where(np.isfinite(comp) == True)[0]
        i = np.argmin((comp[msk] - frac) ** 2.)
        return bins[i]

    def recovered(self, mag_err_lim=None, dx_lim=None):
        """Generates a boolean array indicating if each star is recovered or
        not. This effectively is a boolean AND of results from
        `recovered_in_images`.
        
        A star is recovered if:
            
        1. Recovered magnitude error in any band is less than `mag_err_limit`.
        2. Recovered position is within `dx_lim` pixels of the artificial star.

        and if DOLPHOT observes a star at all at the artificial star's site.

        mag_err_lim : float
            Maximum absolute difference in magnitudes, in any band, for the
            star to be considered recovered.
        dx_lim : float
            Maximum distance between a fake star's input site and its
            observed site for the fake star to be considered recovered.
        """
        if self.nImages > 1:
            recoveryArrays = [self.recovered_in_image(0,
                    mag_err_lim=mag_err_lim, dx_lim=dx_lim)
                    for i in xrange(self.nImages)]
            rec = recoveryArrays[0]
            for r in recoveryArrays[1:]:
                rec = rec & r
            return rec
        else:
            return self.recovered_in_images(0,
                    mag_err_lim=mag_err_lim, dx_lim=dx_lim)

    def recovered_in_image(self, n, mag_err_lim=None, dx_lim=None):
        """Generates a boolean array indicating if each star is recovered in
        the given image (`n`) or not.

        A star is recovered if:
            
        1. Recovered magnitude error in any band is less than `mag_err_limit`.
        2. Recovered position is within `dx_lim` pixels of the artificial star.

        and if DOLPHOT observes a star at all at the artificial star's site.
        
        n : int
            Index of image.
        mag_err_lim : float
            Maximum absolute difference in magnitudes, in any band, for the
            star to be considered recovered.
        dx_lim : float
            Maximum distance between a fake star's input site and its
            observed site for the fake star to be considered recovered.
        """
        if dx_lim is not None:
            k, dx = self.position_errors()
        if self.nImages > 1:
            fakeMag = self.data['fake_mag'][:, n]
            obsMag = self.data['mag'][:, n]
        else:
            fakeMag = self.data['fake_mag']
            obsMag = self.data['mag']
        recovered = np.ones(self.data.shape[0], dtype=np.bool)
        if dx_lim is not None:
            recovered[dx > dx_lim] = 0
        if mag_err_lim is not None:
            magErr = np.sqrt((fakeMag - obsMag) ** 2.)
            recovered[magErr > mag_err_lim] = 0
        return recovered

    def metrics(self, magRange, n, magErrLim=None, dxLim=None):
        """Makes scalar metrics of artificial stars in an image.
        
        For each image, results a tuple (RMS mag error, completeness fraction).
        """
        if dxLim is not None:
            k, dx = self.position_errors()
        if self.nImages > 1:
            fakeMag = self.data['fake_mag'][:, n]
            obsMag = self.data['mag'][:, n]
        else:
            fakeMag = self.data['fake_mag']
            obsMag = self.data['mag']
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
        return rms, comp


class DolphotTable(object):
    """Represents the output from Dolphot in an HDF5 table."""
    def __init__(self, hdfPath):
        super(DolphotTable, self).__init__()
        self.hdfPath = hdfPath
        self._open_hdf()

    def _open_hdf(self):
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

    def add_column(self, colname, coldata, shape=None):
        """Add a column to the photometry table.
        
        The procedure for adding columns to pytables tables is given by
        https://gist.github.com/swarbhanu/1405074
        """
        # TODO handle case of existing column

        self.hdf.close()

        # Open it again in append mode
        fileh = tables.openFile(self.hdfPath, "a")
        # group = fileh.root.tmp_phottable
        table = fileh.root.phot

        # Get a description of table in dictionary format
        descr = table.description._v_colObjects
        descr2 = descr.copy()

        # Add a column to description
        if len(coldata.shape) > 1:
            descr2[colname] = tables.Float64Col(dflt=False,
                    shape=tuple([coldata.shape[1]]))
        else:
            descr2[colname] = tables.Float64Col(dflt=False)

        # Create a new table with the new description
        table2 = fileh.createTable(fileh.root, 'table2', descr2, "A table",
                tables.Filters(1))

        # Copy the user attributes
        table.attrs._f_copy(table2)

        # Fill the rows of new table with default values
        for i in xrange(table.nrows):
            table2.row.append()
        # Flush the rows to disk
        table2.flush()

        # Copy the columns of source table to destination
        for col in descr:
            getattr(table2.cols, col)[:] = getattr(table.cols, col)[:]

        # Fill the new column
        getattr(table2.cols, colname)[:] = coldata

        # Remove the original table
        table.remove()

        # Move table2 to table
        table2.move('/','phot')

        # Finally, close the file
        fileh.close()

        # Re-open in read-only mode
        self._open_hdf()

    def set_metadata(self, key, value):
        """Write metadata to the photometry table.
        
        Parameters
        ----------
        key : str
            Key for metadata dict.
        value : 
            Value of metadata.
        """
        self.hdf.close()

        # Open it again in append mode
        fileh = tables.openFile(self.hdfPath, "a")
        # group = fileh.root.tmp_phottable
        table = fileh.root.phot
        setattr(table._v_attrs, key, value)
        table.flush()
        fileh.close()
        self._open_hdf()


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
