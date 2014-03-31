#!/usr/bin/env python
# encoding: utf-8
"""
Tests fo the PhotTable class.
"""
import os

# from astropy.utils.data import download_file
import astropy.table

from delphinus import PhotTable, FakeTable

PHOTPATH = None
FITSPATH = None
AST1 = None
AST2 = None


def setup_module(module):
    """Download test data to setup module."""
    # module.FITSPATH = download_file("http://files.jonathansick.ca/"
    #         "delphinus_test/M31-sky-28_J_fw100k.fits",
    #         cache=True)
    # module.PHOTPATH = download_file("http://files.jonathansick.ca/"
    #         "delphinus_test/M31-sky-28",
    #         cache=True)
    # module.AST1 = download_file("http://files.jonathansick.ca/"
    #         "delphinus_test/M31-sky-28_1.fake",
    #         cache=True)
    # module.AST2 = download_file("http://files.jonathansick.ca/"
    #         "delphinus_test/M31-sky-28_2.fake",
    #         cache=True)
    module.FITSPATH = "/Users/jsick/code/delphinus/test_data/M31-sky-28_J_fw100k.fits"
    module.PHOTPATH = "/Users/jsick/code/delphinus/test_data/M31-sky-28"
    module.AST1 = "/Users/jsick/code/delphinus/test_data/M31-sky-28_1.fake"
    module.AST2 = "/Users/jsick/code/delphinus/test_data/M31-sky-28_2.fake",


class TestPhotTable(object):
    h5path = "test_phottable.hdf5"

    def setup_class(self):
        """Read the phot file."""
        self.phot_table = PhotTable.read_phot(PHOTPATH,
                n_images=2,
                image_names=['J', 'Ks'],
                bands=['J', 'Ks'],
                fits_path=FITSPATH,
                meta=None)
        self.fake_table = FakeTable.read_phot(AST1,
                n_images=2,
                image_names=['J', 'Ks'],
                bands=['J', 'Ks'],
                fits_path=FITSPATH)

    def teardown_class(self):
        if os.path.exists(self.h5path):
            os.remove(self.h5path)

    def test_nimages(self):
        assert self.phot_table.n_images == 2

    def test_error_estimation(self):
        """docstring for test_error_estimation"""
        self.phot_table.estimate_ast_errors(self.fake_table,
                mag_err_lim=0.5, dx_lim=3.)
        assert "comp" in self.phot_table.keys()
        assert "ast_mag_err_0" in self.phot_table.keys()
        assert "ast_mag_err_1" in self.phot_table.keys()

    def test_read_write_hdf5(self):
        """docstring for test_read_write_hdf5"""
        self.phot_table.write(self.h5path, path='phot', format="hdf5")
        new_table = astropy.table.Table.read(self.h5path, path='phot')
        # make sure the metadata looks right after I/O through HDF5
        assert new_table.meta['n_images'] == self.phot_table.n_images
