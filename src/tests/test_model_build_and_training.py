import unittest
from src.training_script.model_build_and_training import build, train
import os

img_width, img_height = 288, 432
model = build(img_width=img_width, img_height=img_height)


class TestModelbuild(unittest.TestCase):

    def test_train(self):
        train(model=model, img_width=img_width,
        img_height=img_height, train_data_path="./datasets/train_srkw",
        validation_data_path="./datasets/val_srkw", no_of_epochs=3)
        file=os.path.isfile("srkw_cnn.h5") 
        self.assertEqual(file, True)


if __name__ == '__main__':
    unittest.main()
