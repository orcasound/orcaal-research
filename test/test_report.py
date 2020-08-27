import unittest
import report 
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os


class TestPreprocess(unittest.TestCase):

    def test_classification_report_sklearn(self):

        cm, true_classes, specificity, sensitivity, class_labels, acc = report.classification_report_sklearn("preprocess_mag_scipy_Srkws.h5", "test")
        self.assertEqual(cm[0,0], 76)
        self.assertEqual(cm[0,1], 25)
        self.assertEqual(cm[1,1], 86)

        self.assertEqual(true_classes[0],0)
        self.assertEqual(acc, 0.8059701492537313)
        self.assertEqual(sensitivity, 0.7524752475247525 )
        self.assertEqual(specificity, 0.8600)


        self.assertEqual(class_labels[0], "calls")
        self.assertEqual(class_labels[1], "nocalls")
