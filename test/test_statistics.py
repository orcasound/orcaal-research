from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import tensorflow as tf
import statistics
import unittest
import random


img_width, img_height = 288, 432
test_datagen = ImageDataGenerator(rescale=1. / 55)
test_data_generator = test_datagen.flow_from_directory(
        "test",
        target_size=(img_width, img_height),
        batch_size=32,
        shuffle=False)

true_classes = test_data_generator.classes

randomlist = []
for j in range(201):
    randomlist.append(random.randint(0, 1))


model = tf.keras.models.load_model("preprocess_mag_scipy_Srkws.h5")
predictions = model.predict_proba(test_data_generator)

class TestPreprocess(unittest.TestCase):

    def test_plot_roc_curve(self):

        lr_auc = statistics.plot_roc_curve("preprocess_mag_scipy_Srkws.h5", "test", predictions, true_classes, randomlist)
        self.assertEqual(lr_auc, 0.857029702970297)
