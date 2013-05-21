#!/usr/bin/env python
# encoding: utf-8
"""
DOLPHOT diagnostics presented in Luminosity-functions.
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec


def make_diagnostic_plot(dolphotTable, i, imageKey, band, fmt, plotPath,
        magLim=None):
    print dolphotTable.image_paths
    nImages = len(dolphotTable.photTable.attrs.image_paths)
    objtype = dolphotTable.photTable.read(field='type')
    if nImages > 1:
        mag = dolphotTable.photTable.read(field='mag')[:, i]
        quality = dolphotTable.photTable.read(field='quality')[:, i]
        chi = dolphotTable.photTable.read(field='chi')[:, i]
        sn = dolphotTable.photTable.read(field='sn')[:, i]
        sharp = dolphotTable.photTable.read(field='sharp')[:, i]
        crowding = dolphotTable.photTable.read(field='crowding')[:, i]
        ecc = dolphotTable.photTable.read(field='ecc')[:, i]
    else:
        mag = dolphotTable.photTable.read(field='mag')
        quality = dolphotTable.photTable.read(field='quality')
        chi = dolphotTable.photTable.read(field='chi')
        sn = dolphotTable.photTable.read(field='sn')
        sharp = dolphotTable.photTable.read(field='sharp')
        crowding = dolphotTable.photTable.read(field='crowding')
        ecc = dolphotTable.photTable.read(field='ecc')

    if magLim is None:
        magBins = np.arange(mag.min(), mag.max(), 0.5)
    else:
        magBins = np.arange(magLim[0], magLim[1], 0.5)
    inds = np.digitize(mag, magBins)
    nBins = inds.max()
    
    magFcn = _measure_lf(mag, inds, nBins)
    chiMean, chiStd = _measure_luminosity_trend(chi, inds, nBins)
    objTypeFractions = _measure_object_types(objtype, inds, nBins)
    flagFrequencies = _measure_flag_frequencies(quality, inds, nBins)
    snMean, snStd = _measure_luminosity_trend(sn, inds, nBins)
    sharpMean, sharpStd = _measure_luminosity_trend(sharp, inds, nBins)
    crowdingMean, crowdingStd = _measure_luminosity_trend(crowding, inds,
            nBins)
    eccMean, eccStd = _measure_luminosity_trend(ecc, inds, nBins)

    fig = Figure(figsize=(7, 10))
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(8, 1, left=0.15, right=0.95, bottom=0.05, top=0.98,
        wspace=None, hspace=None, width_ratios=None, height_ratios=None)

    axMag = fig.add_subplot(gs[0])
    axMag.semilogy(magBins, magFcn, 'k')
    axMag.set_ylabel(r"$N(M)$")
    for tl in axMag.get_xmajorticklabels():
        tl.set_visible(False)
    axMag.set_xlim(magLim)

    axType = fig.add_subplot(gs[1])
    axType.plot(magBins, objTypeFractions[1],
            ls='-', c='k', label="1, Good")
    axType.plot(magBins, objTypeFractions[2],
            ls='-', c='r', label="2, Faint")
    axType.plot(magBins, objTypeFractions[3],
            ls='-', c='g', label="3, Elongated")
    axType.plot(magBins, objTypeFractions[4],
            ls='-', c='b', label="4, Sharp")
    axType.plot(magBins, objTypeFractions[5],
            ls='-', c='c', label="5, Extended")
    axType.set_ylabel(r"Type")
    for tl in axType.get_xmajorticklabels():
        tl.set_visible(False)
    axType.set_xlim(magLim)

    axFlags = fig.add_subplot(gs[2])
    axFlags.plot(magBins, flagFrequencies[0],
            ls='-', c='k', label="0, Good")
    axFlags.plot(magBins, flagFrequencies[1],
            ls='-', c='r', label="1, Ap. Off Chip")
    axFlags.plot(magBins, flagFrequencies[2],
            ls='-', c='g', label="2, Many Bad Pix")
    axFlags.plot(magBins, flagFrequencies[4],
            ls='-', c='b', label="4, Sat at Centre")
    axFlags.plot(magBins, flagFrequencies[8],
            ls='-', c='c', label="8, V. Bad")
    axFlags.set_ylabel(r"Flag")
    for tl in axFlags.get_xmajorticklabels():
        tl.set_visible(False)
    axFlags.set_xlim(magLim)

    axSN = fig.add_subplot(gs[3])
    axSN.errorbar(magBins, snMean, yerr=snStd, color='k')
    axSN.set_ylabel(r"SN")
    for tl in axSN.get_xmajorticklabels():
        tl.set_visible(False)
    axSN.set_xlim(magLim)
    axSN.set_ylim(0., 200)

    axChi = fig.add_subplot(gs[4])
    axChi.errorbar(magBins, chiMean, yerr=chiStd, color='k')
    axChi.set_ylabel("chi")
    for tl in axChi.get_xmajorticklabels():
        tl.set_visible(False)
    axChi.set_xlim(magLim)
    axChi.set_ylim(0., 50)

    axSharp = fig.add_subplot(gs[5])
    axSharp.errorbar(magBins, sharpMean, yerr=sharpStd, color='k')
    axSharp.set_ylabel("sharp")
    for tl in axSharp.get_xmajorticklabels():
        tl.set_visible(False)
    axSharp.set_xlim(magLim)

    axCrowd = fig.add_subplot(gs[6])
    axCrowd.errorbar(magBins, crowdingMean, yerr=crowdingStd, color='k')
    axCrowd.set_ylabel(r"crowd")
    for tl in axCrowd.get_xmajorticklabels():
        tl.set_visible(False)
    axCrowd.set_xlim(magLim)

    axEcc = fig.add_subplot(gs[7])
    axEcc.errorbar(magBins, eccMean, yerr=eccStd, color='k')
    axEcc.set_ylabel(r"Ecc")
    axEcc.set_xlabel(r"mag (%s)" % band)
    axEcc.set_xlim(magLim)

    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure(plotPath, format=fmt)


def _measure_lf(mags, inds, nBins):
    """Measure number of stars in each magnitude bin"""
    n = np.zeros(nBins)
    for i in xrange(nBins):
        w = np.where(inds == i)[0]
        n[i] = len(w)
    return n


def _measure_object_types(objtypes, inds, nBins):
    """In each bins, measures the proportion of objects of each type."""
    classProp = {i: np.nan * np.zeros(nBins) for i in xrange(1, 6)}
    for i in xrange(nBins):
        w = np.where(inds == i)[0]
        n = len(w)
        if n == 0: continue
        bincounts = np.bincount(objtypes[w])[1:]
        for j, bincount in enumerate(bincounts):
            classProp[j + 1][i] = float(bincount) / n
    return classProp


def _measure_flag_frequencies(flags, inds, nBins):
    """Measure the frequency of flags occuring in each bin."""
    flagProps = {i: np.nan * np.zeros(nBins) for i in [0, 1, 2, 4, 8]}
    for i in xrange(nBins):
        w = np.where(inds == i)[0]
        n = len(w)
        if n == 0: continue
        flagProps[0][i] = len(np.where(flags[w] == 0)[0]) / float(n)
        for j in [1, 2, 4, 8]:
            x = len(np.where(j & flags[w] > 0)[0])
            flagProps[j][i] = x / float(n)
    return flagProps


def _measure_luminosity_trend(x, inds, nBins):
    """Measure mean and sigma of x in each bin"""
    mean = np.zeros(nBins)
    sigma = np.zeros(nBins)
    for i in xrange(nBins):
        w = np.where(inds == i)[0]
        mean[i] = x[w].mean()
        sigma[i] = x[w].std()
    return mean, sigma
