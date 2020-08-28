import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'active_learning_script')))
from active_learning import predict
import numpy.testing as npt


class TestPreprocess(unittest.TestCase):

    def test_active_learning(self):
        probability_last, class_ten = predict("srkw_cnn.h5", "../datasets/test_srkw/calls/")
        self.assertEqual(class_ten, 0)
        npt.assert_almost_equal(probability_last, 0.5000879, decimal=3)


if __name__ == '__main__':
    unittest.main()
