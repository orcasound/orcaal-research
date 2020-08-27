import logging
import argparse
import random
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_roc_curve(test_path,
                   model_path,
                   predictions,
                   true_classes,
                   randomlist):
    """ Plot the ROC curve

    Args:
        test_path: The path to the test directory
        model_path: The path to the model
        predictions: The predictions on the test dataset predicted by the model
        true_classes: The true labels of the test set
        randomlist: A numpy array with random integers consisting either 0 or 1

    Returns:
            None
    """
    ns_probs = randomlist
    lr_probs = predictions[:, 0]
    ns_auc = roc_auc_score(true_classes, ns_probs)
    lr_auc = roc_auc_score(true_classes, lr_probs)
    print('Random Classifier: ROC AUC=%.3f' % (ns_auc))
    print('CNN Classifier: ROC AUC=%.3f' % (lr_auc))
    ns_fpr, ns_tpr, _ = roc_curve(true_classes, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(true_classes, lr_probs)
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Random_Classifier')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='CNN_Classifier')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.show()


def main(args):
    model_path = args.modelpath
    test_path = args.testpath
    img_width, img_height = 288, 432

    test_datagen = ImageDataGenerator(rescale=1. / 55)
    test_data_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_width, img_height),
        batch_size=32,
        shuffle=False)

    true_classes = test_data_generator.classes
    # generate  class dataset

    randomlist = []
    for j in range(201):
        randomlist.append(random.randint(0, 1))

    model = tf.keras.models.load_model(model_path)
    predictions = model.predict_proba(test_data_generator)


 plot_roc_curve(model_path, test_path, predictions, true_classes, randomlist)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Predict which images are orcas")
    parser.add_argument(
        '-m',
        '--modelpath',
        type=str,
        help='path to saved model weights',
        required=True)
    parser.add_argument(
        '-c',
        "--testpath",
        type=str,
        help='directory with Test images',
        required=True)

    args = parser.parse_args()

    main(args)
