import unittest
import model_predict


class Testpredict(unittest.TestCase):
    def test_predict(self):
        no_of_predicted_calls, shape = model_predict.predict("preprocess_mag_scipy_Srkws.h5", "content/datasets/val_srkw/calls")
        self.assertEqual(no_of_predicted_calls, 44)
        self.assertEqual(shape, (44, 1))


if __name__ == '__main__'
    unittest.main()
