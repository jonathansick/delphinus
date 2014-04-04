#!/usr/bin/env python
# encoding: utf-8
"""
Dolphot artificial star test data table.
"""

import os

import numpy as np
import astropy
from astropy.table import Table

from delphinus.table.phot import PhotTable
from delphinus.table.readers import FakeReader


class FakeTable(PhotTable):
    """A Dolphot .fake table."""
    def __init__(self, *args, **kw):
        super(FakeTable, self).__init__(*args, **kw)

    @classmethod
    def read_phot(cls, phot_path, n_images=None, image_names=None, bands=None,
            fits_path=None, meta=None):
        """Import the Dolphot .fake file as an astropy Table.
        
        Parameters
        ----------
        phot_path : str
            Filepath for Dolphot .fake output.
        n_images : int, optional
            Number of images expected in photometry file.
        image_names : list, optional
            Sequence of name (string identifiers) for images in photometry.
        bands : list, optional
            Sequence of bandpass names for images in photometry.
        fits_path : str
            Filepath to a reference image that defines the WCS for stellar
            coordinates.
        meta : str
            Optional dictionary that will be added to the table's `meta`
            attribute.

        Returns
        -------
        cls : :class:`FakeTable`
            An instance of :class:`FakeTable`.
        """
        # Parse text into columns
        if n_images is None:
            if image_names is not None:
                n_images = len(image_names)
            elif bands is not None:
                n_images = len(bands)
            else:
                assert False
        reader = FakeReader(phot_path, n_images, ref_image_path=fits_path)
        if meta is None:
            meta = {}
        else:
            meta = dict(meta)
        if image_names is not None:
            meta['image_names'] = image_names
        if bands is not None:
            meta['bands'] = bands
        meta['n_images'] = n_images
        instance = cls(reader.data, meta=meta)
        return instance

    @classmethod
    def _construct_subclass_reader_wrapper(cls, function):
        def new_function(*args, **kwargs):
            return cls(function(*args, **kwargs))
        if function.__doc__:
            new_function.__doc__ = function.__doc__ \
                + "\nWrapped programmatically to return " + cls.__name__
        return new_function

    @classmethod
    def _register_subclass_io(cls, parent_class=astropy.table.Table):
        reader_registrations = []
        # Look at all the existing readers and if they are
        # registered for the parent class then record the name
        # and function they use
        for (name, parent), reader in astropy.io.registry._readers.items():
            if parent_class == parent:
                reader_registrations.append((name, reader))
        # register all those functions for the new class too,
        # except that we need to wrap the function to return an instance
        # of our class instead
        for (name, reader) in reader_registrations:
            new_reader = cls._construct_subclass_reader_wrapper(reader)
            astropy.io.registry.register_reader(name, cls, new_reader)
        # Now do exactly the same for the writers, except no wrapping needed
        writer_registrations = []
        #Get existing
        for (name, parent), writer in astropy.io.registry._writers.items():
            if parent_class == parent:
                writer_registrations.append((name, writer))
        # register new
        for (name, writer) in writer_registrations:
            astropy.io.registry.register_writer(name, cls, writer)

    def mag_errors(self, n=None, name=None, band=None):
        """Compute output-input magnitude difference for AST.

        Returns
        -------
        An nImage by nStars array of output-input differences.
        """
        diffs = self.image_col('mag', n=n, name=name, band=band) \
                - self.image_col('fake_mag', n=n, name=name, band=band)
        return diffs

    def position_errors(self, n=None, name=None, band=None):
        """Prototype for computing position errors for AST as the
        Euclidean distance between input and output (x,y) coordinates.

        .. todo:: Should cache results
        """
        inputX = self['fake_x']
        inputY = self['fake_y']
        obsX = self['x']
        obsY = self['y']
        dx = np.hypot(inputX - obsX, inputY - obsY)
        return dx

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
                    mag_err_lim=mag_err_lim, dx_lim=dx_lim, qcfunc=qcfunc)
                    for i in xrange(self.n_images)]
            rec = recoveryArrays[0]
            for r in recoveryArrays[1:]:
                rec = rec & r
            return rec
        else:
            return self.recovered_in_image(0,
                    mag_err_lim=mag_err_lim, dx_lim=dx_lim, qcfunc=qcfunc)

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
        fake_mag = self.image_col('fake_mag', n=n)
        obs_mag = self.image_col('mag', n=n)
        recovered = np.ones(len(self), dtype=np.bool)
        if qcfunc is not None:  # apply quality control callback
            sel = qcfunc(self, n)
            recovered = recovered & sel
        if dx_lim is not None:
            dx = self.position_errors()
            recovered[dx > dx_lim] = 0
        if mag_err_lim is not None:
            magErr = np.sqrt((fake_mag - obs_mag) ** 2.)
            recovered[magErr > mag_err_lim] = 0
        return recovered

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

        Returns
        -------
        bins : ndarray
            Magnitude bins
        comp : ndarray
            Completeness fraction in each magnitude bin.
        """
        recovered = self.recovered_in_image(n, mag_err_lim=mag_err_lim,
                dx_lim=dx_lim, qcfunc=qcfunc)
        fake_mag = self.image_col('fake_mag', n)
        bins = np.arange(fake_mag.min(), fake_mag.max(), dmag)
        inds = np.digitize(fake_mag, bins)
        rec = np.bincount(inds, weights=recovered, minlength=None)
        tot = np.bincount(inds, weights=None, minlength=None)
        comp = rec / tot
        return bins, comp[1:]

    def completeness_limits(self, mag_err_lim=None, dx_lim=None,
            frac=0.5, dmag=0.1, qcfunc=None):
        """Compute the completeness limit against each image.

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

        Returns
        -------
        comps : list
            List of completeness limits corresponding to each image.
        """
        comps = []
        for n in xrange(self.n_images):
            c = self.completeness_limit_for_image(n,
                    mag_err_lim=mag_err_lim, dx_lim=dx_lim,
                    frac=frac, dmag=dmag, qcfunc=qcfunc)
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
                mag_err_lim=mag_err_lim, dx_lim=dx_lim, dmag=dmag,
                qcfunc=qcfunc)
        # Solve where completeness crosses the threshold
        # TODO this could be made a lot smarter (i.e., ensure completeess
        # is declining; interpolate between bins; smooth)
        msk = np.where(np.isfinite(comp) == True)[0]
        i = np.argmin((comp[msk] - frac) ** 2.)
        return bins[i]

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
        d = np.empty(len(self), dtype=np.dtype(dt))
        d['ra'][:] = self['fake_ra'][:]
        d['dec'][:] = self['fake_dec'][:]
        fmt = {'ra': '%+10.8f', 'dec': '%+10.7f'}
        for i in xrange(self.n_images):
            mtag = 'mag_%i' % i
            dtag = 'dmag_%i' % i
            d[mtag][:] = self.image_col('fake_mag', i)  # input magnitude
            d[dtag][:] = self.mag_errors(i)  # input - obs
            fmt[mtag] = '%+6.3f'
            fmt[dtag] = '%+6.3f'
            # Find obvious drop-outs
            dropout = np.where(np.abs(d[dtag]) > 10.)[0]
            d[dtag][dropout] = 9.99  # label for StarFISH
        # Apply quality selection criteria wrt each image, sequentially
        for i in xrange(self.n_images):
            dtag = 'dmag_%i' % i
            if round_lim is not None:
                rnd = self.image_col('round', i)
                s = np.where((rnd < min(round_lim[i]))
                        | (rnd > max(round_lim[i])))[0]
                d[dtag][s] = 9.99
            if sharp_lim is not None:
                shp = self.image_col('sharp', i)
                s = np.where((shp < min(sharp_lim[i]))
                    | (shp > max(sharp_lim[i])))[0]
                d[dtag][s] = 9.99
            if crowd_lim is not None:
                cwd = self.image_col('crowding', i)
                s = np.where((cwd < min(crowd_lim[i]))
                    | (cwd > max(crowd_lim[i])))[0]
                d[dtag][s] = 9.99
            if qcfunc is not None:
                s = qcfunc(self, i)
                d[dtag][s] = 9.99
        t = Table(d)

        dirname = os.path.dirname(output_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        t.write(output_path,
                format='ascii.fixed_width_no_header',
                delimiter_pad=None,
                bookend=False,
                formats=fmt,
                delimiter=' ')


FakeTable._register_subclass_io()
