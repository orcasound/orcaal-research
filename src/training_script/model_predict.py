import shutil
import os
import logging
import argparse
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import img_to_array

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
    folder_path = test_path
    model_path = model_path

    img_width, img_height = 288, 432

    model = load_model(model_path)
    N = sum(len(files) for _, _, files in os.walk(folder_path))
    data = np.empty((N, img_width, img_height, 3), dtype=np.uint8)

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

        os.makedirs("calls", exist_ok=True)
        if classes[i][0] == 0:
            shutil.copy(f_n, 'calls')
    logger.info(
        f"Detected {sum(len(files) for _, _, files in os.walk('calls'))} srkw calls")


def main(args):
    model_path = args.modelpath
    test_path = args.testpath
    predict(model_path, test_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Predict which images are srkws")
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
        help='directory with PreProcessed images',
        required=True)

    args = parser.parse_args()

    main(args)
