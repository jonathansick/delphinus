#!/usr/bin/env python
# encoding: utf-8
"""
Interface to dolphot [1]_ and related code for reading DOLPHOT outputs.

Note that the documentation here may be copied verbatim from Andrew Dolphin's
documentation where appropriate (e.g. for defining parameters).

.. [1] DOLPHOT. http://americano.dolphinsim.com/dolphot/
"""

import os
import subprocess

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


class Dolphot(object):
    """docstring for Dolphot

    Parameters
    ----------
    workDir : str
        Work directory for photometry
    **params : kwargs
        Parameters general to a run of Dolphot. Image-specific parameters are
        passed with the :meth:`add_image` and :meth:`add_reference` methods.
    
    Attributes
    ----------
    images : list
        List of dictionaries for images being photometered. Has fields for
        parameters, etc, associated with each photometered image
    referenceImage : dict
        Parameter info for reference image. Has fields `path` and `params`.
        Set to None if no reference image is being used
    params : dict
        General parameters (those not specific to images)
    workDir : str
        Work directory for photometry
    outputName : str
        Base name of the output files. Created when calling :meth:`run`.
    apcorPath : str
        Path to the aperture correction data file
    columnsPath : str
        Path to the columns definition file
    infoPath : str
        Path to the info file
    psfsPath : str
        Path to the psfs file
    tablePath : str
        Path to the HDF5 table, if it it create. `None` otherwise.
    """
    def __init__(self, workDir, **params):
        self.images = []
        self.referenceImage = None
        self.params = params
        if os.path.exists(workDir) is False:
            os.makedirs(workDir)
        self.workDir = workDir

    def add_image(self, imagePath, key=None, **params):
        """Add an image to the set to be photometered.
        
        Parameters
        ----------
        imagePath : str
            Full path to the FITS file to be photomered. MEF or single
            extension fits are valid.
        key : str
            String that uniquely identifies an image in your pipeline. The
            key is used in the HDF5 photometry table to show the order
            of images. Set to the file name, without directory and extension
            if left as None.
        **params : kwargs
            Parameters to be passed for this image. See parameter listing
            of :meth:`DolphotParameters.setup_image`.
        """
        if key is None:
            key = os.path.splitext(os.path.basename(imagePath))[0]
        self.images.append({"path": imagePath, "image_key": key,
            "params": params})
    
    def add_reference(self, imagePath, **params):
        """Add an image to the set to be photometered.
        
        Parameters
        ----------
        imagePath : str
            Full path to the FITS file to be photomered. MEF or single
            extension fits are valid.
        **params : dict
            Parameters to be passed for this image. See parameter listing
            of :meth:`DolphotParameters.setup_image`.
        """
        self.referenceImage = {"path": imagePath, "params": params}

    def change_param(self, key, param):
        """Change or add a single parameter. These must be general parameters.
        """
        self.params[key] = param

    def write_parameters(self, outputName):
        """Write parameters to a .params file for DOLPHOT. This method is
        automatically called by :meth:`run`.
        """
        dolphotParams = DolphotParameters(**self.params)
        for doc in self.images:
            path = doc['path']
            imParams = doc['params']
            dolphotParams.setup_image(path, ref=False, **imParams)
        if self.referenceImage is not None:
            refPath = self.referenceImage['path']
            refParams = self.referenceImage['params']
            dolphotParams.setup_image(refPath, ref=True, **refParams)
        self.paramPath = os.path.join(self.workDir, outputName + ".params")
        dolphotParams.write_parameters(self.paramPath)

    def run(self, outputName, clean=True):
        """Run dolphot photometry given the parameter settings."""
        outputPath = os.path.join(self.workDir, outputName)
        self.write_parameters(outputName)
        command = "dolphot %s -p%s" % (outputPath, self.paramPath)
        print command
        subprocess.call(command, shell=True)
        # Define paths to dolphot outputs
        self._define_output_paths(outputName)

    def _define_output_paths(self, outputName):
        """Define paths to the dolphot outputs"""
        self.apcorPath = os.path.join(self.workDir, outputName + ".apcor")
        self.columnsPath = os.path.join(self.workDir, outputName + ".columns")
        self.infoPath = os.path.join(self.workDir, outputName + ".info")
        self.psfsPath = os.path.join(self.workDir, outputName + ".psfs")
        self.photPath = os.path.join(self.workDir, outputName)
        self.outputName = outputName

    def compile_hdf5(self, tablePath=None):
        """Make an HDF5 file of the photometric output."""
        if tablePath is None:
            tablePath = os.path.join(self.workDir, self.outputName + ".h5")
        dolphotTable = DolphotTable.make(tablePath, self.images,
                self.referenceImage, self.photPath, self.psfsPath,
                self.apcorPath)
        self.tablePath = tablePath
        return dolphotTable


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
            FakeStars=None, FakeMatch=3., FakeStarPSF=0,
            RandomFake=1,
            xytfile=None, xytpsf=None):
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
            "RandomFake": RandomFake,
            "xytfile": xytfile, "xytpsf": xytpsf}

    def setup_image(self, path, psfA=(3, 0, 0, 0, 0, 0),
            psfB=(3, 0, 0, 0, 0, 0), psfC=(0, 0, 0, 0, 0, 0),
            shift=(0, 0), xform=(1, 0, 0), aprad=20, apsky=(30, 50),
            raper=2.5, rsky0=4.0, rsky1=10.0, rchi=None, rpsf=15,
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
        rsky0 : real
            Inner radius for computing sky values, if FitSky=1 is being used.
            This should be outside the bulk of the light from the star.
        rsky1 : real
            Outer radius for computing sky values, if FitSky=1 is being used.
            This should be sufficiently large to compute an accurate sky.
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
                "aprad": aprad, "apsky": apsky, "RSky0": rsky0,
                "Rsky1": rsky1, "RAper": raper, "RChi": rchi,
                "RPSF": rpsf, }
        if ref == True:
            self.refImageParams = imageDoc
        else:
            self.imageParams.append(imageDoc)

    def write_parameters(self, path):
        """Write the parameter file to `path`."""
        pathDir = os.path.dirname(path)
        if os.path.exists(pathDir) is False: os.makedirs(path)
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
                "RSky0": "%.2f",
                "Rsky1": "%.2f",
                "RAper": "%.2f",
                "RChi": "%.2f",
                "RPSF": "%i"}
        lines = []
        for key, p in params.iteritems():
            if p is None: continue
            if n == 0 and key in ['shift', 'xform']: continue
            if key not in formatters: continue  # don't know this key
            print key, p, formatters[key]
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
            "FakeStarPSF": "%i", "RandomFake": "%i",
            "xytfile": "%s", "xytpsf": "%s"}
        lines = []
        for key, p in self.params.iteritems():
            if p is None: continue
            if key not in formatters: continue  # don't know this key
            print key, p, formatters[key]
            lines.append(key + " = " + formatters[key] % p)
        return lines


class DolphotTable(object):
    """Represents the output from Dolphot in an HDF5 table."""
    def __init__(self, tablePath):
        super(DolphotTable, self).__init__()
        self.tablePath = tablePath
    
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


"""
1. Extension (zero for base image)
2. Chip (for three-dimensional FITS image)
3. Object X position on reference image (or first image, if no reference)
4. Object Y position on reference image (or first image, if no reference)
5. Chi for fit
6. Signal-to-noise
7. Object sharpness
8. Object roundness
9. Direction of major axis (if not round)
10. Crowding
11. Object type (1=bright star, 2=faint, 3=elongated, 4=hot pixel, 5=extended)
12. Measured counts, image 1
13. Measured sky level, image 1
14. Normalized count rate, image 1
15. Normalized count rate uncertainty, image 1
16. Instrumental magnitude, image 1
17. Magnitude uncertainty, image 1
18. Chi, image 1
19. Signal-to-noise, image 1
20. Sharpness, image 1
21. Roundness, image 1
22. Crowding, image 1
23. PSF FWHM, image 1
24. PSF eccentricity, image 1
25. PSF a parameter, image 1
26. PSF b parameter, image 1
27. PSF c parameter, image 1
28. Photometry quality flag, image 1
29. Measured counts, image 2
30. Measured sky level, image 2
31. Normalized count rate, image 2
32. Normalized count rate uncertainty, image 2
33. Instrumental magnitude, image 2
34. Magnitude uncertainty, image 2
35. Chi, image 2
36. Signal-to-noise, image 2
37. Sharpness, image 2
38. Roundness, image 2
39. Crowding, image 2
40. PSF FWHM, image 2
41. PSF eccentricity, image 2
42. PSF a parameter, image 2
43. PSF b parameter, image 2
44. PSF c parameter, image 2
45. Photometry quality flag, image 2
"""