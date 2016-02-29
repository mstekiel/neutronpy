# -*- coding: utf-8 -*-
from neutronpy import form_facs
import numpy as np
import unittest
from mock import patch
from matplotlib import use
use('Agg')


class StructureFactor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(StructureFactor, self).__init__(*args, **kwargs)

        self.input = {'name': 'FeTe',
                      'composition': [{'ion': 'Fe', 'pos': [0.75, 0.25, 0.]},
                                      {'ion': 'Fe', 'pos': [1. - 0.75, 1. - 0.25, 0.0]},
                                      {'ion': 'Te', 'pos': [0.25, 0.25, 1. - 0.2839]},
                                      {'ion': 'Te', 'pos': [1. - 0.25, 1. - 0.25, 1. - (1. - 0.2839)]}],
                      'debye-waller': True,
                      'massNorm': True,
                      'formulaUnits': 1.,
                      'lattice': dict(abc=[3.81, 3.81, 6.25], abg=[90, 90, 90])}

    def test_str_fac(self):
        structure = form_facs.Material(self.input)
        self.assertAlmostEqual(np.abs(structure.calc_str_fac((2., 0., 0.))) ** 2, 1702170.4663405998, 6)
        self.assertAlmostEqual(np.abs(structure.calc_str_fac((2, 0, 0))) ** 2, 1702170.4663405998, 6)
        self.assertAlmostEqual(np.abs(structure.calc_str_fac((0, 2., 0))) ** 2, 1702170.4663405998, 6)
        self.assertAlmostEqual(np.abs(structure.calc_str_fac((0, 2, 0))) ** 2, 1702170.4663405998, 6)

        ndarray_example = np.linspace(0.5, 1.5, 21)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((ndarray_example, 0, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((0, ndarray_example, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((0, 0, ndarray_example))) ** 2), 16831011.814390473, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((ndarray_example, ndarray_example, 0))) ** 2), 10616602.544519115, 6)


        list_example = list(ndarray_example)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((list_example, 0, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((0, list_example, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((0, 0, list_example))) ** 2), 16831011.814390473, 6)

        tuple_example = tuple(ndarray_example)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((tuple_example, 0, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((0, tuple_example, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((0, 0, tuple_example))) ** 2), 16831011.814390473, 6)

    def test_N_atoms(self):
        structure = form_facs.Material(self.input)
        self.assertTrue(structure.N_atoms(22) == 36110850351331465494528)

    def test_volume(self):
        structure = form_facs.Material(self.input)
        self.assertTrue(structure.volume == 90.725624999999965)

    def test_total_scattering_cross_section(self):
        structure = form_facs.Material(self.input)
        self.assertTrue(structure.total_scattering_cross_section == 31.880000000000003)

    def test_case(self):
        input_test = self.input
        del input_test['formulaUnits']
        structure = form_facs.Material(input_test)

    @patch("matplotlib.pyplot.show")
    def test_plot(self, mock_show):
        structure = form_facs.Material(self.input)
        structure.plot_unit_cell()

    def test_optimal_thickness(self):
        structure = form_facs.Material(self.input)
        self.assertEqual(structure.calc_optimal_thickness(), 1.9552936422413782)


class MagneticFormFactor(unittest.TestCase):
    def test_mag_form_fac(self):
        ion = form_facs.Ion('Fe')
        formfac, _temp = ion.calc_mag_form_fac(q=1.)[0], ion.calc_mag_form_fac(q=1.)[1:]
        self.assertAlmostEqual(formfac, 0.932565, 6)

    def test_mag_form_fac_case1(self):
        ion = form_facs.Ion('Fe')
        formfac, _temp = ion.calc_mag_form_fac()[0], ion.calc_mag_form_fac()[1:]
        self.assertAlmostEqual(np.sum(formfac), 74.155233575216599, 12)

    def test_mag_form_fac_case2(self):
        ion = form_facs.Ion('Fe')
        formfac, _temp = ion.calc_mag_form_fac(qrange=[0, 2])[0], ion.calc_mag_form_fac(qrange=[0, 2])[1:]
        self.assertAlmostEqual(np.sum(formfac), 74.155233575216599, 12)


if __name__ == "__main__":
    unittest.main()
