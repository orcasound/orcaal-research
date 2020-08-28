import unittest
from active_learning_script import active_learning
import numpy.testing as npt


class TestPreprocess(unittest.TestCase):

    def test_active_learning(self):
        probability_last, class_ten = active_learning.predict("srkw_cnn.h5", "datasets/test_srkw/calls/")
        self.assertEqual(class_ten, 0)
        npt.assert_almost_equal(probability_last, 0.5279, decimal=3)


if __name__ == '__main__':
    unittest.main()
