#!/usr/bin/env python
# encoding: utf-8
"""
Readers for Dolphot photometry and AST tables.
"""
import os

import numpy as np
import numpy.lib.recfunctions as recf

from astropy.wcs import WCS
import astropy.io.fits
from astropy.table import Table


class BasePhotReader(object):
    """Base class for reading Dolphot photometry output files.

    Parameters
    ----------
    filepath : str
        Path to the DOPHOT '.phot' file.
    n_images : int
        Number of imagse in the dataset.
    ref_image_path : str
        (Optional) path to the reference FITS image. Used to define the
        WCS system to compute sky coordinates of stars. Overriden by
        ``ref_wcs_list``.
    ref_wcs_list : list
        (Optional) ``list`` of :class:`astropy.wcs.WCS` instances for each
        image extension. Numbers are zero-based and must correspond to order
        of extensions in the `.phot` file.

    Attributes
    ----------
    filepath : str
        Path to the DOPHOT '.phot' file.
    n_images : int
        Number of imagse in the dataset.
    data : :class:`numpy.ndarray`
        Record array baed on DOLPHOT '.phot' file.
    """
    N_IMAGE_COLS = 17
    N_GLOBAL_COLS = 11
    GLOBAL_COL_OFFSET = 0

    def __init__(self, filepath, n_images,
                 ref_image_path=None,
                 ref_wcs_list=None):
        super(BasePhotReader, self).__init__()
        self.filepath = filepath
        self.n_images = n_images
        self._idata = None
        self._gdata = None
        self.data = None

        self.ref_image_path = ref_image_path
        if ref_wcs_list is not None:
            self.ref_wcs_list = ref_wcs_list
        elif ref_image_path is not None:
            self.ref_wcs_list = self._build_ref_wcs(ref_image_path)
        else:
            self.ref_wcs_list = None

        self._read()

    def extract_additional_columns(self, data, n_images, nStars):
        """User hook for extracting additional columns from the photometry
        output file.
        """
        pass

    def _build_ref_wcs(self, ref_path):
        """Make a list of :class:`astropy.wcs.WCS` instances for each extension
        of the reference image.
        """
        wcs_list = []
        with astropy.io.fits.open(ref_path) as f:
            for ext in f:
                wcs = WCS(ext.header)
                wcs_list.append(wcs)
        return wcs_list

    def _read(self):
        """Pipeline for reading DOLPHOT photometry output."""
        data = np.loadtxt(self.filepath)
        nstars = data.shape[0]
        self._extract_global_cols(data, self.GLOBAL_COL_OFFSET, nstars)
        self._extract_image_phot_cols(
            data,
            self.GLOBAL_COL_OFFSET + self.N_GLOBAL_COLS,
            self.n_images, nstars)
        self.extract_additional_columns(data, self.n_images, nstars)  # hook
        self._fill_radec()
        self.combine_structured_array()
        self._idata = None
        self._gdata = None

    def _extract_image_phot_cols(self, data, offset, n_images, nStars):
        """Extract output for image-specific photometry columns."""
        number = lambda name, i: name + "_{0}".format(str(i))
        col_names = (
            'counts', 'sky', 'norm_count_rate', 'norm_count_rate_err',
            'mag', 'mag_err', 'chi', 'sn', 'sharp', 'round', 'crowding',
            'fwhm', 'ecc', 'psf_a', 'psf_b', 'psf_c', 'quality')
        dt = []
        for i in xrange(n_images):
            dt.extend([(number(n, i), np.float) for n in col_names])
        self._idata = np.empty(nStars, dtype=np.dtype(dt))
        for i in range(self.n_images):
            for j, name in enumerate(col_names):
                k = offset + i * self.N_IMAGE_COLS + j
                self._idata[number(name, i)] = data[:, k]

    def _extract_global_cols(self, data, offset, nStars):
        """Extract output for global image columns."""
        dt = [('ext', np.int),
              ('z', np.int),  # 1 for simple 2D images
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
        col_names = [d[0] for d in dt]
        self._gdata = np.empty(nStars, dtype=np.dtype(dt))
        for i, name in enumerate(col_names):
            j = i + offset
            self._gdata[name] = data[:, j]

    def _fill_radec(self, ra_col='ra', dec_col='dec', x_col='x', y_col='y',
                    table=None):
        """Compute RA/Dec columns, if a reference WCS is specified."""
        if table is None:
            table = self._gdata
        if self.ref_wcs_list is not None:
            # Compute coordinates for each extension
            for ext, wcs in enumerate(self.ref_wcs_list):
                s = np.where(table['ext'] == ext)[0]
                ra, dec = wcs.all_pix2world(table[x_col][s],
                                            table[y_col][s],
                                            1)
                table[ra_col] = ra
                table[dec_col] = dec
        else:
            table[ra_col][:] = np.nan
            table[dec_col][:] = np.nan

    def combine_structured_array(self):
        """docstring for combine_structured_array"""
        arrays = [self._gdata, self._idata]
        self.data = recf.merge_arrays(arrays, flatten=True, usemask=False)

    @property
    def table(self):
        """Make an astropy table from the read dataset."""
        tbl = Table(data=self.data)
        return tbl


class FakeReader(BasePhotReader):
    """Read Dolphot's .fake artificial star output."""
    N_FAKE_GLOBAL_COLS = 4
    N_FAKE_IMAGE_COLS = 2

    def __init__(self, filepath, n_images, ref_image_path=None):
        self.GLOBAL_COL_OFFSET = self.N_FAKE_GLOBAL_COLS \
            + n_images * self.N_FAKE_IMAGE_COLS
        super(FakeReader, self).__init__(
            filepath, n_images,
            ref_image_path=ref_image_path)

    def __add__(self, other):
        """Return a concatenated FakeReader (concatenates data)."""
        self.data = np.concatenate((self.data, other.data))
        return self

    def extract_additional_columns(self, data, n_images, nStars):
        """Reads additional columns for .fake output.

        Called from the superclass BasePhotReader's _read() method.
        """
        self._extract_fake_global_cols(data, nStars)
        self._extract_fake_phot_cols(data, n_images, nStars)

    def _extract_fake_global_cols(self, data, nStars):
        """Extract global columns at beginning of .fake files."""
        dt = [('fake_ext', np.int),
              ('fake_z', np.int),  # 1 for 2D images
              ('fake_x', np.float),
              ('fake_y', np.float),
              ('fake_ra', np.float),
              ('fake_dec', np.float)]
        self._fgdata = np.empty(nStars, dtype=np.dtype(dt))
        for i in range(self.N_FAKE_GLOBAL_COLS):
            colname = dt[i][0]
            self._fgdata[colname] = data[:, i]
        self._fake_fill_radec()

    def _extract_fake_phot_cols(self, data, n_images, nStars):
        """Extract input photometry columns at beginning of .fake files
        for each image.
        """
        number = lambda name, i: name + "_{0}".format(str(i))
        col_names = ['fake_count', 'fake_mag']
        dt = []
        for i in xrange(n_images):
            dt.extend([(number(n, i), np.float) for n in col_names])
        self._fidata = np.empty(nStars, dtype=np.dtype(dt))
        for i in range(self.n_images):
            for j, name in enumerate(col_names):
                k = self.N_FAKE_GLOBAL_COLS + i * self.N_FAKE_IMAGE_COLS + j
                self._fidata[number(name, i)] = data[:, k]

    def _fake_fill_radec(self):
        """Compute RA/Dec columns, if a reference image is specified."""
        self._fill_radec(
            ra_col='fake_ra', dec_col='fake_dec',
            x_col='fake_x', y_col='fake_y', table=self._fgdata)

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
        if self.n_images > 1:
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
                     frac=0.5, dmag=0.1, qcfunc=None):
        """Returns magnitude vs completeness fraction for the given image.

        Parameters
        ----------

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
        qcfunc :
            Callback function for applying quality cuts while assessing
            completeness.
        """
        recovered = self.recovered_in_image(
            n, mag_err_lim=mag_err_lim,
            dx_lim=dx_lim, qcfunc=qcfunc)
        if self.n_images > 1:
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
                            frac=0.5, dmag=0.1, qcfunc=None):
        """Compute the completeness limit against each image.
        Returns a list of completeness limits corresponding to each image.

        Parameters
        ----------
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
        qcfunc :
            Callback function for applying quality cuts while assessing
            completeness.
        """
        comps = []
        for n in xrange(self.n_images):
            c = self.completeness_limit_for_image(
                n,
                mag_err_lim=mag_err_lim,
                dx_lim=dx_lim,
                frac=frac,
                dmag=dmag,
                qcfunc=qcfunc)
            comps.append(c)
        return comps

    def completeness_limit_for_image(self, n, mag_err_lim=None, dx_lim=None,
                                     frac=0.5, dmag=0.1, qcfunc=None):
        """Compute the completeness limit against each a single image.

        Parameters
        ----------
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
        qcfunc :
            Callback function for applying quality cuts while assessing
            completeness.
        """
        bins, comp = self.completeness(n,
                                       mag_err_lim=mag_err_lim,
                                       dx_lim=dx_lim,
                                       dmag=dmag,
                                       qcfunc=qcfunc)
        # Solve where completeness crosses the threshold
        # TODO this could be made a lot smarter (i.e., ensure completeess
        # is declining; interpolate between bins; smooth)
        msk = np.where(np.isfinite(comp) == True)[0]  # NOQA
        i = np.argmin((comp[msk] - frac) ** 2.)
        return bins[i]

    def recovered(self, mag_err_lim=None, dx_lim=None, qcfunc=None):
        """Generates a boolean array indicating if each star is recovered or
        not. This effectively is a boolean AND of results from
        :meth:`recovered_in_image`.

        A star is recovered if:

        1. Recovered magnitude error in any band is less than `mag_err_limit`.
        2. Recovered position is within `dx_lim` pixels of the artificial star.

        and if DOLPHOT observes a star at all at the artificial star's site.

        Parameters
        ----------

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
        if self.n_images > 1:
            recoveryArrays = [self.recovered_in_image(0,
                                                      mag_err_lim=mag_err_lim,
                                                      dx_lim=dx_lim,
                                                      qcfunc=qcfunc)
                              for i in xrange(self.n_images)]
            rec = recoveryArrays[0]
            for r in recoveryArrays[1:]:
                rec = rec & r
            return rec
        else:
            return self.recovered_in_image(0,
                                           mag_err_lim=mag_err_lim,
                                           dx_lim=dx_lim,
                                           qcfunc=qcfunc)

    def recovered_in_image(self, n, mag_err_lim=None, dx_lim=None,
                           qcfunc=None):
        """Generates a boolean array indicating if each star is recovered in
        the given image (`n`) or not.

        A star is recovered if:

        1. Recovered magnitude error in any band is less than `mag_err_limit`.
        2. Recovered position is within `dx_lim` pixels of the artificial star.

        and if DOLPHOT observes a star at all at the artificial star's site.

        Parameters
        ----------
        n : int
            Index of image.
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
        if dx_lim is not None:
            k, dx = self.position_errors()
        if self.n_images > 1:
            fakeMag = self.data['fake_mag'][:, n]
            obsMag = self.data['mag'][:, n]
        else:
            fakeMag = self.data['fake_mag']
            obsMag = self.data['mag']
        recovered = np.ones(self.data.shape[0], dtype=np.bool)
        if qcfunc is not None:  # apply quality control callback
            sel = qcfunc(self.data, n)
            recovered = recovered & sel
        if dx_lim is not None:
            recovered[dx > dx_lim] = 0
        if mag_err_lim is not None:
            magErr = np.sqrt((fakeMag - obsMag) ** 2.)
            recovered[magErr > mag_err_lim] = 0
        return recovered

    def export_for_starfish(self, output_path, round_lim=None, sharp_lim=None,
                            crowd_lim=None, qcfunc=None):
        """Export artificial star test data for the StarFISH `synth` command.

        Parameters
        ----------
        output_path : str
            Path where crowding table will be written.
        qcfunc :
            Callback function for applying quality cuts while assessing
            completeness.
        """
        dt = [('ra', np.float), ('dec', np.float)]
        for i in xrange(self.n_images):
            dt.append(('mag_%i' % i, np.float))
            dt.append(('dmag_%i' % i, np.float))
        d = np.empty(self.data.shape[0], dtype=np.dtype(dt))
        d['ra'][:] = self.data['fake_ra'][:]
        d['dec'][:] = self.data['fake_dec'][:]
        fmt = {'ra': '%+10.8f', 'dec': '%+10.7f'}
        for i in xrange(self.n_images):
            mtag = 'mag_%i' % i
            dtag = 'dmag_%i' % i
            d[mtag][:] = self.data['fake_mag'][:, i]  # input magnitude
            d[dtag][:] = self.data['mag'][:, i] - self.data['fake_mag'][:, i]
            fmt[mtag] = '%+6.3f'
            fmt[dtag] = '%+6.3f'
            # Find obvious drop-outs
            dropout = np.where(np.abs(d[dtag]) > 10.)[0]
            d[dtag][dropout] = 9.99  # label for StarFISH
        # Apply quality selection criteria wrt each image, sequentially
        for i in xrange(self.n_images):
            dtag = 'dmag_%i' % i
            if round_lim is not None:
                s = np.where(
                    (self.data['round'][:, i] < min(round_lim[i]))
                    | (self.data['round'][:, i] > max(round_lim[i])))[0]
                d[dtag][s] = 9.99
            if sharp_lim is not None:
                s = np.where(
                    (self.data['sharp'][:, i] < min(sharp_lim[i]))
                    | (self.data['sharp'][:, i] > max(sharp_lim[i])))[0]
                d[dtag][s] = 9.99
            if crowd_lim is not None:
                s = np.where(
                    (self.data['crowding'][:, i] < min(crowd_lim[i]))
                    | (self.data['crowding'][:, i] > max(crowd_lim[i])))[0]
                d[dtag][s] = 9.99
            if qcfunc is not None:
                s = qcfunc(self.data, i)
                d[dtag][s] = 9.99
        dirname = os.path.dirname(output_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        t = Table(d)
        t.write(output_path,
                format='ascii.fixed_width_no_header',
                delimiter_pad=None,
                bookend=False,
                formats=fmt,
                delimiter=' ')

    def metrics(self, magRange, n, magErrLim=None, dxLim=None):
        """Makes scalar metrics of artificial stars in an image.

        For each image, results a tuple (RMS mag error, completeness fraction).

        TODO deprecated. Replace with other methods.
        """
        if dxLim is not None:
            k, dx = self.position_errors()
        if self.n_images > 1:
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
