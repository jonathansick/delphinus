#!/usr/bin/env python
# encoding: utf-8
"""
Demonstration for artificial star test reduction.

2013-07-05 - Created by Jonathan Sick
"""

from glob import glob

from delphinus.artstars import ASTReducer
from delphinus.phottable import FakeReader, DolphotTable


def main():
    h5path = "/Users/jsick/androphot_test/M31-sky-28/M31-sky-28.h5"
    fakepaths = glob("/Users/jsick/androphot_test/M31-sky-28/M31-sky-28*.fake")
    refImgPath = "/Users/jsick/androphot_test/M31-sky-28/M31-sky-28_J_nightset.fits"
    phot_tbl = DolphotTable(h5path)
    fake_tbls = [FakeReader(p, 2, refImagePath=refImgPath) for p in fakepaths]
    fake_tbl = fake_tbls[0] + fake_tbls[1]
    reducer = ASTReducer(fake_tbl, phot_tbl)
    print "Completeness limits:"
    print reducer.completeness_limits(mag_err_lim=0.2)
    print "Compute errors"
    reducer.compute_errors(mag_err_lim=0.2, dx_lim=3) 

if __name__ == '__main__':
    main()
