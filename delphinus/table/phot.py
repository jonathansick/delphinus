#!/usr/bin/env python
# encoding: utf-8
"""
Dolphot photometry table reader/writer.
"""
import os

import numpy as np
import astropy.table.table
from astropy.table import Column

from delphinus.table.readers import BasePhotReader
from delphinus.ast import estimate_errors


class PhotTable(astropy.table.table.Table):
    """A DolphotPhotometry table."""
    def __init__(self, *args, **kw):
        super(PhotTable, self).__init__(*args, **kw)

    @classmethod
    def read_phot(cls, phot_path, n_images=None, image_names=None, bands=None,
            fits_path=None, meta=None):
        """Import the Dolphot photometry file as an astropy Table.
        
        Parameters
        ----------
        phot_path : str
            Filepath for Dolphot photometry output.
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
        cls : :class:`PhotTable`
            An instance of :class:`PhotTable`.
        """
        # Parse text into columns
        if n_images is None:
            if image_names is not None:
                n_images = len(image_names)
            elif bands is not None:
                n_images = len(bands)
            else:
                assert False
        reader = BasePhotReader(phot_path, n_images, ref_image_path=fits_path)
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
        # Get existing
        for (name, parent), writer in astropy.io.registry._writers.items():
            if parent_class == parent:
                writer_registrations.append((name, writer))
        # register new
        for (name, writer) in writer_registrations:
            astropy.io.registry.register_writer(name, cls, writer)

    @property
    def image_names(self):
        """Names of images."""
        return self._read_meta('image_names')

    @property
    def image_bands(self):
        """Bandpasses of images."""
        return self._read_meta('bands')

    @property
    def n_images(self):
        """Count the nmber of images in photometry."""
        return self._read_meta('n_images')

    def _read_meta(self, key):
        """Astropy Table FITS I/O does a weird thing with randomly scambling
        the case of metadata keywords. This method reads metadata, trying
        different cases for the keyword."""
        if key.lower() in self.meta:
            return self.meta[key.lower()]
        if key.upper() in self.meta:
            return self.meta[key.upper()]
        else:
            return None

    def _resolve_index(self, n=None, name=None, band=None):
        """Figure out what image we're talking about."""
        if n is not None:
            N = int(n)
        elif name is not None:
            N = self.image_names.index(name)
        elif band is not None:
            N = self.image_bands.index(band)
        else:
            # Bold assumption
            N = 0
        return N

    def image_column_name(self, key, n=None, name=None, band=None):
        """Get the name for an image-specfific column."""
        N = self._resolve_index(n=n, name=name, band=band)
        col_name = "{0}_{1:d}".format(key, N)
        return col_name

    def image_col(self, key, n=None, name=None, band=None):
        """Read data for an image-specific column. The image can either be
        resolved from a number, name, or bandpass (in order of preference).
        """
        col_name = self.image_column_name(key, n=n, name=name, band=band)
        return self[col_name]

    def estimate_ast_errors(self, fake_tbl,
            mag_err_lim=None, dx_lim=None, qcfunc=None):
        """Incorporate artificial star tests for error estimation.

        This method adds the following columns to the table:

        - `comp`: the completeness probability for each star
        - `ast_mag_err_N`: the magnitude uncertainty in the `N` image (for
          each image).
        
        Parameters
        ----------
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
        """
        comps, sigmas = estimate_errors(self, fake_tbl,
                mag_err_lim=mag_err_lim, dx_lim=dx_lim, qcfunc=qcfunc)
        comp_col = Column(name='comp', data=comps)
        self.add_column(comp_col)
        for i in xrange(self.n_images):
            sigma_col = Column(name="ast_mag_err_{0}".format(str(i)),
                data=sigmas[:, i])
            self.add_column(sigma_col)

    def export_for_starfish(self, output_path, xaxis, yaxis,
            xspan, yspan, sel=None, apcor=None):
        """Create a photometric catalog that can be directly used by
        StarFISH, a tool for CMD decompositions and star formation history
        analysis.

        Parameters
        ----------
        output_path : str
            Path where the photometry will be saved. In StarFISH terminology,
            this path should be ``datpre + suffix``.
        xaxis : int or length-2 tuple
            Specify the magnitude system of the x-axis in the StarFISH CMD
            according to indices of filters in the photometry catalog.
            If ``xaxis`` is an ``int``, then that magnitude is the ``xaxis``.
            If ``xaxis`` is a tuple, then the x-axis is computed as the
            difference (colour) of the two indices in the tuple.
        yaxis : int or tuple
            Same as ``xaxis``, but for the y-axis.
        xspan : tuple
            Span of stellar magnitudes on the x-axis that will be exported.
            This must match the span of the CMD space being used by StarFISH.
        yspan : tuple
            Span of stellar magnitudes on the y-axis that will be exported.
        sel : ndarray
            Index into the photometry table, giving indices of stars
            to select (include in the export). If left as `None`, then
            only the default selections for location on the CMD will be
            made.
        apcor : ndarray
            Aperture corrections that will be *added* to the instrumental
            magnitudes during export. ``apcor`` should be the length of the
            number of magnitudes.
        """
        mags = np.column_stack([self.image_col.read('mag', n)
            for n in xrange(self.n_images)])
        for i in xrange(self.n_images):  # apply aperture corrections
            mags[:, i] += apcor[i]
        if isinstance(xaxis, int):
            if sel is None:
                xdata = mags[:, xaxis]
            else:
                xdata = mags[sel, xaxis]
        else:
            if sel is None:
                xdata = mags[:, xaxis[0]] - mags[:, xaxis[1]]
            else:
                xdata = mags[sel, xaxis[0]] - mags[sel, xaxis[1]]
        if isinstance(yaxis, int):
            if sel is None:
                ydata = mags[:, yaxis]
            else:
                ydata = mags[sel, yaxis]
        else:
            if sel is None:
                ydata = mags[:, yaxis[0]] - mags[:, yaxis[1]]
            else:
                ydata = mags[sel, yaxis[0]] - mags[sel, yaxis[1]]
        # Filter stars that appear in CMD plane
        s = np.where((xdata > min(xspan)) & (xdata < max(xspan))
                & (ydata > min(yspan)) & (ydata < max(yspan)))[0]
        xdata = xdata[s]
        ydata = ydata[s]
        nstars = len(xdata)

        dt = [('xaxis', np.float), ('yaxis', np.float)]
        data = np.empty(nstars, dtype=np.dtype(dt))
        data['xaxis'] = xdata
        data['yaxis'] = ydata
        t = astropy.table.table.Table(data)

        dirname = os.path.dirname(output_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        t.write(output_path,
                format='ascii.fixed_width_no_header',
                delimiter_pad=None,
                bookend=False,
                formats={'xaxis': "%6.3f", 'yaxis': "%6.3f"},
                delimiter=' ')


PhotTable._register_subclass_io()
