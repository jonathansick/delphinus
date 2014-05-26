#!/usr/bin/env python
# encoding: utf-8
"""
Interface to dolphot [1]_ and related code for reading DOLPHOT outputs.

Note that the documentation here may be copied verbatim from Andrew Dolphin's
documentation where appropriate (e.g. for defining parameters).

.. [1] DOLPHOT. http://americano.dolphinsim.com/dolphot/
"""

import os
import shutil
import time
import subprocess

from .table import PhotTable
from .table import FakeTable


class Dolphot(object):
    """The :class:`Dolphot` class manages a single run of `dolphot` itself.

    This class manages `dolphot` parameters, runs `dolphot`, and provides
    paths to the `dolphot` output, including the ability to compile the
    `dolphot` photometry tables into a Pythonic data structure.

    Parameters
    ----------
    work_dir : str
        Work directory for photometry
    **params : kwargs
        Parameters general to a run of Dolphot. Image-specific parameters are
        passed with the :meth:`add_image` and :meth:`add_reference` methods.
    
    Attributes
    ----------
    images : list
        List of dictionaries for images being photometered. Has fields for
        parameters, etc, associated with each photometered image
    ref_image : dict
        Parameter info for reference image. Has fields `path` and `params`.
        Set to None if no reference image is being used
    params : dict
        General parameters (those not specific to images)
    work_dir : str
        Work directory for photometry
    output_name : str
        Base name of the output files. Created when calling :meth:`run`.
    exec_time : float
        Execution time (seconds) of most recent Dolphot run.
    param_path : str
        Path of most recently written Dolphot parameters file.
    fake_path : str
        Path to .fake output file.
    """
    def __init__(self, work_dir, **params):
        self.images = []
        self.ref_image = None
        self.params = dict(params)
        if os.path.exists(work_dir) is False:
            os.makedirs(work_dir)
        self.work_dir = work_dir
        self.exec_time = None
        self.param_path = None
        self.fake_path = None

        self._phot_table = None  # PhotTable of current Dolphot run
        self._fake_table = None  # FakeTable of current Dolphot run

    def add_image(self, image_path, key=None, band=None, **params):
        """Add an image to the set to be photometered.
        
        Parameters
        ----------
        image_path : str
            Full path to the FITS file to be photomered. MEF or single
            extension fits are valid.
        key : str
            String that uniquely identifies an image in your pipeline. The
            key is used in the HDF5 photometry table to show the order
            of images. Set to the file name, without directory and extension
            if left as None.
        band : str
            Band-pass of this image. Can be used to help create cursory
            CMDs. If not set, the bandpass of each image is labeled `'None'`.
        **params : kwargs
            Parameters to be passed for this image. See parameter listing
            of :meth:`DolphotParameters.setup_image`.
        """
        if key is None:
            key = os.path.splitext(os.path.basename(image_path))[0]
        if band is None:
            band = 'None'
        self.images.append({"path": image_path, "image_key": key,
            "params": dict(params), "band": band})
    
    def add_reference(self, image_path, **params):
        """Add an image to the set to be photometered.
        
        Parameters
        ----------
        image_path : str
            Full path to the FITS file to be photomered. MEF or single
            extension fits are valid.
        **params : dict
            Parameters to be passed for this image. See parameter listing
            of :meth:`DolphotParameters.setup_image`.
        """
        self.ref_image = {"path": image_path, "params": dict(params)}

    def change_param(self, key, param):
        """Change or add a single parameter. These must be general parameters.
        """
        self.params[key] = param

    def write_parameters(self, output_name, full_path=False):
        """Write parameters to a .params file for DOLPHOT. This method is
        automatically called by :meth:`run`.
        """
        dol_params = DolphotParameters(**self.params)
        for doc in self.images:
            dol_params.setup_image(doc['path'], ref=False, **doc['params'])
        if self.ref_image is not None:
            dol_params.setup_image(self.ref_image['path'], ref=True,
                    **self.ref_image['params'])
        if not full_path:
            self.param_path = os.path.join(self.work_dir,
                    output_name + ".params")
        else:
            self.param_path = output_name
        dol_params.write_parameters(self.param_path)

    def run(self, output_name, clean=True):
        """Run dolphot photometry given the parameter settings."""
        self.output_name = output_name
        output_path = os.path.join(self.work_dir, output_name)
        self.write_parameters(output_name)
        command = "dolphot %s -p%s" % (output_path, self.param_path)
        with Timer() as t:
            subprocess.call(command, shell=True)
        self.exec_time = t.interval
        self._phot_table = None  # reset
        _fake_path = output_path + ".fake"
        if os.path.exists(_fake_path):
            self.fake_path = _fake_path

    def run_ast(self, output_name, starlist_name):
        """Run dolphot photometry in artificial star test mode.
        
        This method differs from :meth:`run` in that it will create a
        directory to perform the AST in, so that multiple star lists
        can be simultaneously run on a single input photometry. That is,
        this method prevents the AST output of one star list from clobbering
        that of another.
        """
        self.output_name = output_name
        ast_dir = os.path.join(self.work_dir,
                "%s_%s" % (output_name, starlist_name))
        if not os.path.exists(ast_dir):
            os.makedirs(ast_dir)
        # Symlink outputs into ast_dir
        _apcor_path = os.path.join(ast_dir, output_name + ".apcor")
        _cols_path = os.path.join(ast_dir, output_name + ".columns")
        _info_path = os.path.join(ast_dir, output_name + ".info")
        _psfs_path = os.path.join(ast_dir, output_name + ".psfs")
        _fake_path = os.path.join(ast_dir, output_name + ".fake")
        _phot_path = os.path.join(ast_dir, output_name)
        os.symlink(self.apcor_path, _apcor_path)
        os.symlink(self.info_path, _info_path)
        os.symlink(self.columns_path, _cols_path)
        os.symlink(self.psfs_path, _psfs_path)
        os.symlink(self.phot_path, _phot_path)
        # Run photometry
        self.write_parameters(os.path.join(ast_dir, output_name + ".params"),
            full_path=True)
        outputPath = os.path.join(ast_dir, output_name)
        command = "dolphot %s -p%s" % (outputPath, self.param_path)
        with Timer() as t:
            subprocess.call(command, shell=True)
        self.exec_time = t.interval
        # Move AST result back to main directory, labeled with star list name
        self.fake_path = os.path.join(self.work_dir,
                "%s_%s.fake" % (output_name, starlist_name))
        if os.path.exists(self.fake_path):
            os.remove(self.fake_path)
        shutil.move(_fake_path, self.fake_path)
        # Delete AST dir
        shutil.rmtree(ast_dir)
        self._fake_table = None  # reset

    @property
    def phot_path(self):
        """Path to Dolphot photometry output table."""
        return os.path.join(self.work_dir, self.output_name)

    @property
    def apcor_path(self):
        """Path to .apcor output file."""
        return os.path.join(self.work_dir, self.output_name + ".apcor")

    @property
    def columns_path(self):
        """Path to .columns output file."""
        return os.path.join(self.work_dir, self.output_name + ".columns")

    @property
    def info_path(self):
        """Path to .info output file."""
        return os.path.join(self.work_dir, self.output_name + ".info")

    @property
    def psfs_path(self):
        """Path to .psfs output file."""
        return os.path.join(self.work_dir, self.output_name + ".psfs")

    @property
    def warnings_path(self):
        """Path to .warnings output file."""
        return os.path.join(self.work_dir, self.output_name + ".warnings")

    @property
    def psf_fits_paths(self):
        """Path list of PSF FITS files."""
        return [os.path.join(self.work_dir,
            self.output_name + ".%i.psf.fits" % n)
            for n in xrange(1, len(self.images) + 1)]

    @property
    def res_fits_paths(self):
        """Paths to residual FITS files."""
        return [os.path.join(self.work_dir,
            self.output_name + ".%i.res.fits" % n)
            for n in xrange(1, len(self.images) + 1)]

    def label_fake_output(self, label):
        """Renames artificial star test output (useful it running multiple
        fake star lists on the same photometry)."""
        newPath = os.path.join(self.work_dir,
                "%s_%s.fake" % (self.output_name, label))
        if newPath == self.fake_path:
            return
        if os.path.exists(newPath):
            os.remove(newPath)
        shutil.move(self.fake_path, newPath)
        self.fake_path = newPath

    @property
    def phot_table(self):
        """A PhotTable with the current results."""
        if self._phot_table is None:
            image_names = [im['image_key'] for im in self.images]
            bands = [im['band'] for im in self.images]
            if self.ref_image is not None:
                ref_image_path = self.ref_image['path']
            else:
                ref_image_path = self.images[0]['path']
            meta = {'apcor': self.apcor_path, 'info': self.info_path,
                    'psfs_fits_path': self.psf_fits_paths,
                    'res_fits_paths': self.res_fits_paths,
                    'exec_time': self.exec_time}
            self._phot_table = PhotTable.read_phot(self.phot_path,
                    n_images=len(self.images),
                    image_names=image_names,
                    bands=bands,
                    fits_path=ref_image_path,
                    meta=meta)
        return self._phot_table

    @property
    def fake_table(self):
        """A FakeTable with the current results."""
        if self._fake_table is None and self.fake_path is not None:
            image_names = [im['image_key'] for im in self.images]
            bands = [im['band'] for im in self.images]
            if self.ref_image is not None:
                ref_image_path = self.ref_image['path']
            else:
                ref_image_path = self.images[0]['path']
            meta = {'exec_time': self.exec_time}
            self._fake_table = FakeTable.read_phot(self.fake_path,
                    n_images=len(self.images),
                    image_names=image_names,
                    bands=bands,
                    fits_path=ref_image_path,
                    meta=meta)
        return self.self._fake_table


class DolphotParameters(object):
    """Handles parameters for Dolphot.
    
    General parameters are initialzed upon instantiation, while parameters
    for individual images are specified in the setup_image command.
    
    Parameters
    ----------
    PSFPhot : int
        Type of photometry to be run. Options are

        - 0 (aperture),
        - 1 (standard PSF-fit)
        - 2 (PSF-fit weighted for central pixels).
        
        Option 1 is suggested for most photometric needs, but option 0 can
        provide superior photometry if the signal-to- noise is high and the
        field is uncrowded.
    FitSky : int
        Sky-fitting setting. Options are
        
        - 0 (use the sky map from calcsky)
        - 1 (fit the sky normally prior to each photometry measurement)
        - 2 (fit the sky inside the PSF region but outside the photometry
          aperture),
        - 3 (fit the sky within the photometry aperture as a
          2-parameter PSF fit), and
        - 4 (fit the sky within the photometry
          aperture as a 4-parameter PSF fit).
        
        Options 1 and 3 are the suggested settings. Option 0 should be used
        only if the field is very uncrowded; option 2 can be used in extremely
        crowded fields; option 4 can help in fields with strong background
        gradients (though I have yet to see this be useful).
    RCentroid : int
        The centroid used for obtaining initial positions of stars is a
        square of size 2RCentroid + 1 on each side.
    SigFind : real
        Sigma detection threshold. Stars above this limit will be kept in the
        photometry until the final output.
    SigFindMult : real
        Multiple for sigma detection threshold in initial finding algorithm.
        This should be close to one for larger PSFs, and as low as 0.75 for
        badly undersampled PSFs.
    SigFinal : real
        Sigma threshold for a star to be listed in the final photometry list.
        To get all stars, set SigFinal equal to SigFind.
    MaxIT : int
        Maximum number of photometry iterations.
    FSPS : str
        The functional form of the analytic PSF. Allowable options are “Gauss”
        (Gaussian), “Lorentz” (Lorentzian), “Lorentz2” (a squared Lorentzian),
        and “G+L” (sum of Gaussian and Lorentzian).
    SkipSky : int
        Sampling of sky annulus; set to a number higher than 1 to gain speed at
        the expense of precision. This is only used if FitSky is set to 1.
        In general, this should never be larger than the FWHM of the PSF.
    SkySig : real
        Sigma rejection threshold for sky fit; only used if FitSky is set to 1.

    NoiseMult : real
        To allow for imperfect PSFs, the noise is increased by this value
        times the star brightness in the pixel when computing chi values.
    FSat : real
        Fraction of nominal saturation for which pixesl are considered
        saturated.
    Zero : real
        Zero point for a star of brightness 1 DN per second.
    PosStep : real
        Typical stepsize in x and y during photometry iterations. Should be set
        to a factor of a few smaller than the PSF FHWM.
    dPosMax : real
        Maximum position change of a star during a single photometry iteration.
        Note that this parameter is currently ignored.
    RCombine : real
        Minimum separation of two stars (they will be combined if they become
        closer than this value). This value can generally be about 2/3 of the
        PSF FWHM, but no less than 1.42.
    sigPSF : real
        Minimum signal-to-noise for a PSF solution to be attempted on a star.
        Fainter detections are assigned type 2.
    PSFStep : real
        Typical stepsize of FWHM during photometry iterations. Setting to zero
        will replace PSF solution with three-state solution in which a star
        will be very small, fit the stellar PSF, or very large.
    MinS : real
        Minimum FWHM for a good star (type 1). This should be set to something
        like half the PSF FWHM.
    MaxS : real
        Maximum FWHM for a good star (type 1). This needs to be set to
        something larger than the FWHM of the PSF.
    MaxE : real
        Maximum ellipticity for a good star (type 1).
    UseWCS : int
        Use WCS header information for alignment? Allowed values are 0 (no),
        1 (use to estimate shift, scale, and rotation), or 2 (use to estimate
        a full distortion solution). Note that any shifts or rotations selected
        by img shift and img xform are applied in addition to what is
        determined by the WCS solution. If reducing HST data, selecting
        UseWCS=1 can eliminate the need for running the fitdistort utilities.
        UseWCS=2 generally is not recommended for HST data since the
        distortion coefficients provided with DOLPHOT provide higher-order
        corrections than do the WCS headers.
    Align : int
        Align images to reference? Allowed values are 0 (no),
        1 (x/y offsets only), 2 (x/y offsets plus scale difference),
        and 3 (x/y offsets plus distortion).
    Rotate : int
        Correct for rotation in alignment? Allowed values are 0 (no) and
        1 (yes).
    SecondPass : int
        Number of additional passes when finding stars to locate stars in the
        wings of brighter stars. Must be a non-negative value.
    SearchMode : int
        Sets the minimization used to determine star positions. Allowed values
        are 0 (chi divided by SNR) and 1 (chi only). A value of one appear
        safe for all applications. A value of zero has been seen to fail if
        images of very different exposure times are used together.
    Force1 : int
        Force all objects to be of class 1 or 2? Allowed values are 0 (no) and
        1 (yes). For crowded stellar fields, this should be set to 1 and the
        chiand sharpness values used to discard extended objects.
    EPSF : int
        Allow elliptical PSFs in parameter fits? Allowed values are 0 (no) and
        1 (yes).
    PSFsol : int
        Make analytic PSF solution? Allowed values are -1 (no),
        0 (constant PSF), 1 (linear PSF variation), and
        2 (quadratic PSF variation).
    PSFres : int
        Solve for PSF residual image? Allowed values are 0 (no) and 1 (yes).
        Turning this feature off can create nonlinearities in the photometry
        unless PSFphot is also set to zero.
    psfstars : str or None
        Specify coordinates of PSF stars. The file must contain extension,
        chip, X, and Y (the first four columns of DOLPHOT output).
    psfoff : real
        Coordinate offset of PSF star list. Values equal the list coordinates
        minus the DOLPHOT coordinates, and would thus be 0.5 if using a DAOPHOT
        or IRAF star list.
    ApCor : int
        Make aperture corrections? Allowed values are 0 (no) and 1 (yes).
        Default aperture corrections always have the potential for error, so
        it is strongly recommended that you manually examine the raw output
        from this process.
    SubPixel : int
        The number of PSF calculations made per dimension per pixel. For
        Nyquist-sampled images, this can be set to 1, but very small PSFs
        require the extra precision.
    FakeStars : str or None
        Run DOLPHOT in artificial star mode. The FakeStars parameter is the
        name of the text file containing the artificial star data. The file
        should contain the following information for each star, one star per
        line: extension (0 = main image), chip (usually 1), X, Y, and the
        number of counts on each image. If the warmstart option is being used,
        one also needs to specify the recovered X, Y, and object type values
        before the counts. Note that photometry must be run first; the
        photometry list, PSFs, etc. from DOLPHOT are used as inputs in the
        fake star routine.
    FakeMatch : real
        Maximum allowable distance between input and recovered artificial star.
    FakePSF : real
        Approximate FWHM of the image, used to determine which of two input
        stars a recovered star should be matched with.
    FakeStarPSF : int
        Use PSF residual from initial photometry run. This should be left at
        zero, unless the PSF residuals are small and well-measured.
    RandomFake : int
        Apply Poisson noise to fake stars when adding them. This should always
        be used, unless running fake star tests twice (once with and once
        without) to quantify photometric errors from crowding and background
        independently of the errors due to photon noise.
    xytfile : str or None
        star list filename for warmstart
    xytpsf : str or None
        PSF solution for the reference image for difference image photomtry.
    photsec : list or None
        Optinally, specify a subset fo the image to be photometered (relative
        to first image, or referene image). Pass a list of six integer values:
        extension (usually 0), Z (usually 1), minimum X, minimum Y, maximum X,
        maximum Y.
    """
    def __init__(self, PSFPhot=2, FitSky=1, SkipSky=1, SkySig=2.25,
            SigFind=2.5, SigFindMult=0.85,
            SigFinal=3.0, MaxIT=25, NoiseMult=0.25, FSat=0.999, Zero=25.0,
            RCentroid=3, PosStep=0.6,
            dPosMax=3.0, RCombine=1.75, FPSF="G+L",
            sigPSF=10, PSFStep=0.25, MinS=1.0, MaxS=9.0, MaxE=0.5,
            UseWCS=0, Align=3, Rotate=1, secondPass=1, SearchMode=1,
            Force1=0,
            EPSF=1,
            PSFsol=0, PSFres=1, psfstars=None, psfoff=0.0,
            ApCor=0, SubPixel=1,
            FakeStars=None, FakeMatch=3., FakeStarPSF=0, FakePSF=1.5,
            RandomFake=1,
            xytfile=None, xytpsf=None, photsec=None):
        self.refImageParams = None
        self.imageParams = []
        self.params = {"PSFPhot": PSFPhot, "FitSky": FitSky,
            "SkipSky": SkipSky,
            "SkySig": SkySig, "SigFind": SigFind, "SigFindMult": SigFindMult,
            "SigFinal": SigFinal, "MaxIT": MaxIT, "NoiseMult": NoiseMult,
            "FSat": FSat, "Zero": Zero,
            "RCentroid": RCentroid, "PosStep": PosStep,
            "dPosMax": dPosMax, "RCombine": RCombine, "FPSF": FPSF,
            "sigPSF": sigPSF, "PSFStep": PSFStep, "MinS": MinS, "MaxS": MaxS,
            "MaxE": MaxE, "UseWCS": UseWCS, "Align": Align, "Rotate": Rotate,
            "secondPass": secondPass, "SearchMode": SearchMode,
            "Force1": Force1, "EPSF": EPSF, "PSFsol": PSFsol,
            "PSFres": PSFres, "psfstars": psfstars, "psfoff": psfoff,
            "ApCor": ApCor, "SubPixel": SubPixel, "FakeStars": FakeStars,
            "FakeMatch": FakeMatch, "FakeStarPSF": FakeStarPSF,
            "FakePSF": FakePSF, "RandomFake": RandomFake,
            "xytfile": xytfile, "xytpsf": xytpsf, "photsec": photsec}

    def setup_image(self, path, psfA=(3, 0, 0, 0, 0, 0),
            psfB=(3, 0, 0, 0, 0, 0), psfC=(0, 0, 0, 0, 0, 0),
            shift=(0, 0), xform=(1, 0, 0), aprad=20, apsky=(30, 50),
            RAper=2.5, RSky=(4.0, 10.0), RChi=None, RPSF=15,
            ref=False):
        """Configure the fitting parameters for a single image. This may
        also be the reference image if `ref=True` is set.
        
        Parameters
        ----------
        path : str
            file path of FITS file (without the .fits suffix)
        psfA : tuple
            PSF XX term, length 6. Set the PSF x-FWHM and linear and quadratic
            variations. This value can be an initial guess that is later
            adjusted by DOLPHOT.
        psfB : tuple
            PSF YY term, length 6. Set the PSF y-FWHM and linear and quadratic
            variations. This value can be an initial guess that is later
            adjusted by DOLPHOT.
        psfC : tuple
            PSF XY term, length 6. Set the PSF eccentricity and linear and
            quadratic variations. This value can be an initial guess that is
            later adjusted by DOLPHOT.
        shift : tuple
            x,y shifts relative to reference. Set offset of image relative to
            reference. This value can be an initial guess that is later
            adjusted by DOLPHOT. Values are x and y on the image minus x and y
            on the reference image. Note that this parameter should not be set
            for the reference image.
        xform : tuple
            Set the scale ratio, cubic distortion, and rotation of the image
            relative to the reference image. This value can be an initial
            guess that is later ad- justed by DOLPHOT. Note that this
            parameter should not be set for the reference image.
        aprad : real
            Radius for aperture correction
        apsky : tuple
            Set the inner and outer radii of the annulus used for calculating
            sky values for aperture corrections.
        raper : real
            Sets the size of the aperture within which photometry will be
            performed. For FitSky=0 or 1, this should include most of the
            light of the star. For FitSky=2, 3, or 4 options, this should also
            include significant “sky” area outside the star.
        rsky : tuple
            Inner, outer radius for computing sky values, if FitSky=1 is being
            used. This should be outside the bulk of the light from the star.
            Outer radius should be sufficiently large to compute an accurate
            sky.
        rpsf : int
            Size of the PSF used for star subtraction. The rule of thumb is to
            make sure this is sufficiently large that significant unsubtracted
            star light is not seen beyond the subtracted regions in the
            residual image.
        rchi : real
            Sets the size of the aperture within which the chi value will be
            calculated. This is used to determine object locations. This
            should generally include only the peak of the stellar PSF. RChi
            cannot be larger than RAper. If not defined, RChi is set equal to
            RAper.
        ref : bool
            set as True if this is the reference image, False otherwise
        """
        # Chop off the .fits extension if necessary
        if path.endswith(".fits"):
            path = os.path.splitext(path)[0]
        imageDoc = {"file": path, "psfa": psfA, "psfb": psfB,
                "psfc": psfC, "shift": shift, "xform": xform,
                "aprad": aprad, "apsky": apsky,
                "RSky": RSky, "RAper": RAper, "RChi": RChi,
                "RPSF": RPSF, }
        if ref == True:
            self.refImageParams = imageDoc
        else:
            self.imageParams.append(imageDoc)

    def write_parameters(self, path):
        """Write the parameter file to `path`."""
        # pathDir = os.path.dirname(path)
        # if os.path.exists(pathDir) is False: os.makedirs(path)
        if os.path.exists(path) is True: os.remove(path)
        paramLines = []
        paramLines.append("Nimg = %i" % len(self.imageParams))
        if self.refImageParams is not None:
            paramLines += self._write_image_params(self.refImageParams, 0)
        for i, imageParam in enumerate(self.imageParams):
            paramLines += self._write_image_params(imageParam, i + 1)
        paramLines += self._write_general_params()
        paramTxt = "\n".join(paramLines) + "\n"
        f = open(path, 'w')
        f.write(paramTxt)
        f.close()

    def _write_image_params(self, params, n):
        """Produces a list of strings, giving parameters for photometry of
        a given image (or reference)
        
        Parameters
        ----------
        d : dict
            Dictionary of parameters for the image
        n : int
            Number of the image. Reference image is 0.
        """
        prefix = "img%i_" % n
        formatters = {"file": "%s",
                "psfa": "%.2f %.2f %.2f %.2f %.2f %.2f",
                "psfb": "%.2f %.2f %.2f %.2f %.2f %.2f",
                "psfc": "%.2f %.2f %.2f %.2f %.2f %.2f",
                "shift": "%.2f %.2f",
                "xform": "%.2f %.2f %.2f",
                "aprad": "%.2f",
                "apsky": "%.2f %.2f",
                "RSky": "%.2f %.2f",
                "RAper": "%.2f",
                "RChi": "%.2f",
                "RPSF": "%i"}
        lines = []
        for key, p in params.iteritems():
            if p is None: continue
            if n == 0 and key in ['shift', 'xform']: continue
            if key not in formatters: continue  # don't know this key
            if type(p) is list:
                p = tuple(p)
            lines.append(prefix + key + " = " + formatters[key] % p)
        return lines

    def _write_general_params(self):
        """Produces a list of strings for the general parameters not associated
        with specific images."""
        formatters = {"PSFPhot": "%i", "FitSky": "%i", "SkipSky": "%.2f",
            "SkySig": "%.2f", "SigFind": "%.2f", "SigFindMult": "%.2f",
            "SigFinal": "%.2f", "MaxIT": "%i", "NoiseMult": "%.2f",
            "FSat": "%.2f", "Zero": "%.2f",
            "RCentroid": "%i", "PosStep": "%.2f",
            "dPosMax": "%.2f", "RCombine": "%.2f", "FPSF": "%s",
            "sigPSF": "%.2f", "PSFStep": "%.2f", "MinS": "%.2f",
            "MaxS": "%.2f",
            "MaxE": "%.2f", "UseWCS": "%i", "Align": "%i", "Rotate": "%i",
            "secondPass": "%i", "SearchMode": "%i",
            "Force1": "%i", "EPSF": "%i", "PSFsol": "%i", "PSFres": "%i",
            "psfstars": "%s", "psfoff": "%.1f", "ApCor": "%i",
            "SubPixel": "%i", "FakeStars": "%s", "FakeMatch": "%.2f",
            "FakeStarPSF": "%i", "FakePSF": "%.2f", "RandomFake": "%i",
            "xytfile": "%s", "xytpsf": "%s",
            "photsec": "%i %i %i %i %i %i"}
        lines = []
        for key, p in self.params.iteritems():
            if p is None: continue
            if key not in formatters: continue  # don't know this key
            if type(p) is list:
                p = tuple(p)
            if key == "photsec":
                print key, p
            lines.append(key + " = " + formatters[key] % p)
        return lines


class Timer:
    """Timer function, via http://preshing.com/
       20110924/timing-your-code-using-pythons-with-statement

    This timer uses :func:`time.time` to measure wall time so that it can
    accurately measure the time taken by subprocesses (such as `dolphot`).
    """
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
