import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'training_script')))
from model_predict import predict


class TestPreprocess(unittest.TestCase):
    def test_predict(self):
        no_of_predicted_calls, shape = predict("srkw_cnn.h5", "../datasets/test_srkw/calls")
        self.assertEqual(no_of_predicted_calls, 3)
        self.assertEqual(shape, (3, 1))


if __name__ == '__main__':
    unittest.main()
