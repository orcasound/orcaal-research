import unittest
import active_learning
import numpy.testing as npt


class TestPreprocess(unittest.TestCase):

    def test_active_learning(self):
        probability_last, class_ten = active_learning.predict("preprocess_mag_scipy_Srkws.h5", "test/calls/")

        self.assertEqual(class_ten, 0)
        npt.assert_almost_equal(probability_last, 0.5279, decimal=3)
