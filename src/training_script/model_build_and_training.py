import argparse
import logging
import logging.config
import os
import tensorflow as tf
from keras import backend as K
from keras import optimizers, regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# Disable PIL.PngImagePlugin DEBUG logs
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": True,
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SRKWs:
    @staticmethod
    def build(img_width, img_height):
        """Build Convolution Neural Network model

        Create a Convolution-Neural-Network model for detection of calls
        and no calls of the Southern Resident Killer Whales.

        Args:
             img_width: The width of the image
             img_height: The height of the image
        Returns:
               model
        """
        if K.image_data_format() == "channels_first":
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

        model = Sequential()
        model.add(Conv2D(64, (5, 5),
                  kernel_regularizer=regularizers.l2(0.001),
                  padding="same",
                  strides=3,
                  input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, (5, 5),
                  kernel_regularizer=regularizers.l2(0.001),
                  strides=3,
                  padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(128, (5, 5),
                  kernel_regularizer=regularizers.l2(0.001),
                  strides=3,
                  padding="same"))
        model.add(Activation("relu"))
        model.add(Dropout(0.6))

        model.add(Conv2D(128, (5, 5),
                  kernel_regularizer=regularizers.l2(0.001),
                  strides=3,
                  padding="same"))
        model.add(Activation("relu"))
        model.add(Dropout(0.6))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))

        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(lr=3e-5),
            metrics=["accuracy"])

        model.summary()
        return model


def train(
          model, img_width, img_height, train_data_path,
          validation_data_path, no_of_epochs):
    """Train the Convolution Neural Network model

    Trains the model on the training dataset which consists of calls and
    no calls.The trained model is also saved with the name srkw_cnn.h5

    Args:
        model: The CNN model that we created
        img_width: The width of the image
        img_height: The height of the image
        train_data_path: The path to the training folder
        validation_data_path: The path to the validation folder
        no_of_epochs: The number of epochs for which we want to train
                      our model
    Returns:
            None
    """
    nb_train_samples = sum(len(files)
                           for _, _, files in os.walk(train_data_path))
    nb_validation_samples = sum(len(files)
                                for _, _, files in os.walk(
                                    validation_data_path))

    epochs = no_of_epochs
    batch_size = 32
    checkpoint = ModelCheckpoint(
            filepath="checkpoint_srkw-{epoch:02d}-{val_loss:.2f}.h5",
            monitor="val_loss", verbose=0, save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1,
                                  patience=100, min_lr=1e-8)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2)

    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_batchsize = 32
    val_batchsize = 22

    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(img_width, img_height),
        batch_size=train_batchsize,
        class_mode="binary",
        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_path,
        target_size=(img_width, img_height),
        batch_size=val_batchsize,
        class_mode="binary",
        shuffle=False)

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint, reduce_lr])

    model.save("srkw_cnn.h5")

    logger.info("Detection Model saved")


def main(args):
    dataset_path = args.classpath
    no_of_epochs = args.noofepochs

    train_data_path = os.path.join(dataset_path, "train_srkw/")
    validation_data_path = os.path.join(dataset_path, "val_srkw/")

    img_width, img_height = 288, 432

    logger.info("Starting compiling of SRKWs ... ")
    model = SRKWs.build(img_width=img_width, img_height=img_height)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizers.Adam(lr=3e-5),
                  metrics=["accuracy"])
    logger.info("Starting Training ... ")
    train(model=model,
          img_width=img_width,
          img_height=img_height,
          train_data_path=train_data_path,
          validation_data_path=validation_data_path,
          no_of_epochs=no_of_epochs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train CNN model for detection of srkws calls in spectrograms")
    parser.add_argument(
        "-c",
        "--classpath",
        type=str,
        help="directory with pos and neg samples in two respective folders",
        required=True)
    parser.add_argument(
        "-epochs",
        "--noofepochs",
        type=int,
        help="Enter the number of epochs for which you want to train",
        default=256
        )
    args = parser.parse_args()

    main(args)
