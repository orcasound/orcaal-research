import unittest
import sys
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join('..', 'training_script')))
from report import classification_report_sklearn

class TestReport(unittest.TestCase):

    def test_classification_report_sklearn(self):

        cm, true_classes, specificity, sensitivity, class_labels, acc = classification_report_sklearn('srkw_cnn.h5', '../datasets/test_srkw/')
        self.assertEqual(cm[0, 0], 3)
        self.assertEqual(cm[1, 1], 0)

        self.assertEqual(true_classes[0], 0)
        self.assertEqual(acc, 0.3333333333333333)
        self.assertEqual(sensitivity, 1.0000)
        self.assertEqual(specificity, 0.0000)


        self.assertEqual(class_labels[0], 'calls')
        self.assertEqual(class_labels[1], 'no_calls')


if __name__ == '__main__':
    unittest.main()
