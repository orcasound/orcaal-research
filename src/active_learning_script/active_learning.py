import shutil
import os
import logging
import argparse
from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import img_to_array
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict(model_path, test_path):
    """Predict the labels using predict function

    Args:
        model_path: The path to the model or the .h5 file
        test_path: The path to the test directory

    Returns:
           None.
    """

    img_width, img_height = 288, 432
    model = tf.keras.models.load_model(model_path)
    folder_path = test_path
    model_path = model_path
    N = sum(len(files) for _, _, files in os.walk(folder_path))
    data = np.empty((N, img_width, img_height, 3), dtype=np.uint8)
    predictions_probab = model.predict_proba(data)
    one_dim_predict = predictions_probab

    for dirs, _, files in os.walk(folder_path):
        for i, file in enumerate(files):
            f_name = os.path.join(dirs, file)
            img = image.load_img(f_name, target_size=(img_width, img_height))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            data[i, ...] = img

    logger.info("Starting Prediction")
    classes = model.predict_classes(data, batch_size=32)

    f = []
    for i in os.listdir(folder_path):
        f.append(i)

    for i in range(len(classes)):
        f_n = os.path.join(folder_path, f[i])

        os.makedirs("uncertain_calls", exist_ok=True)
        if one_dim_predict[i] > 0.1 and one_dim_predict[i] < 0.9:
            shutil.copy(f_n, 'uncertain_calls')
    logger.info(
        f"There are  {sum(len(files) for _, _, files in os.walk('uncertain_calls'))} uncertain srkw calls")
    return one_dim_predict[i][0], classes[10][0]


def main(args):
    model_path = args.modelpath
    test_path = args.testpath
    predict(model_path, test_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Predict which are uncertain_calls")
    parser.add_argument(
        '-m',
        '--modelpath',
        type=str,
        help='path to saved model weights',
        required=True)
    parser.add_argument(
        '-c',
        "--datapath",
        type=str,
        help='directory with Preprocessed images',
        required=True)

    args = parser.parse_args()

    main(args)
