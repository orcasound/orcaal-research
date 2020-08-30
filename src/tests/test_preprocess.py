import unittest
sys.path.append(os.path.abspath(os.path.join('..', 'preprocessing_script')))
from preprocessing_script import preprocess
import librosa
import pandas as pd
import os
import numpy.testing as npt


call_annotations = pd.read_csv("datasets/test_extract_audio.tsv", sep="\t")
call_annotations["end"] = call_annotations["start"] + call_annotations["duration_s"]


class TestPreprocess(unittest.TestCase):

    def test_generate_negative_tsv(self):
        """Test to check the negative_tsv_generated file"""

        negative_tsv_generated = preprocess.generate_negative_tsv(
                             call_annotations, 3,
                             "datasets")

        print(negative_tsv_generated)

        # Testing the name of the first file
        self.assertEqual(negative_tsv_generated.iloc[0][0], "1562337136_0004.wav")

        # Testing the id of the first file
        self.assertEqual(negative_tsv_generated.iloc[0][1], 0)

        # Testing the label of the first file
        self.assertEqual(negative_tsv_generated.iloc[0][4], 0)

        # Testing if the start-time of the call is not contained in .tsv
        self.assertNotEqual(negative_tsv_generated.iloc[0][2], 49.765625)
        self.assertNotEqual(negative_tsv_generated.iloc[1][2], 49.765625)
        self.assertNotEqual(negative_tsv_generated.iloc[2][2], 49.765625)

        # Testing the name of the first file with id '1' and label '0'
        self.assertEqual(negative_tsv_generated.iloc[1][0], "1562337136_0004.wav")
        self.assertEqual(negative_tsv_generated.iloc[1][1], 1)
        self.assertEqual(negative_tsv_generated.iloc[1][4], 0)

    def test_extract_audio(self):
        preprocess.extract_audio(
                   ".",
                   "datasets/", 3, call_annotations)


        # Testing the number of samples generated are as expected

        data_test, sr_test = librosa.core.load("datasets/test_wav.wav", res_type="kaiser_best")
        data, sr = librosa.core.load("datasets/extracted_calls1.wav", res_type="kaiser_best", sr=22050)

        # Check the call produced and the sampling rate is same as the expected call
        self.assertEqual(data.shape, data_test.shape)
        self.assertEqual(sr, sr_test)
        npt.assert_almost_equal(data, data_test, decimal=1)

    def test_apply_per_channel_energy_norm(self):
        """ Test to check the PCEN spectrograms """
        data_test, sr_test = librosa.core.load("datasets/test_wav.wav", res_type="kaiser_best", sr=22050)
        data, sr = librosa.core.load("datasets/extracted_calls1.wav", res_type="kaiser_best", sr=22050)

        pcen_spectrogram = preprocess.apply_per_channel_energy_norm(data)
        pcen_spectrogram_test = preprocess.apply_per_channel_energy_norm(data_test)
        npt.assert_almost_equal(
                    pcen_spectrogram, pcen_spectrogram_test, decimal=1)

    def test_wavelet_denoising(self):
        """ Test to check Wavelet denoised spectrograms """
        data_test, sr_test = librosa.core.load("datasets/test_wav.wav", res_type="kaiser_best", sr=22050)
        data, sr = librosa.core.load(
                   "datasets/extracted_calls1.wav",
                   res_type="kaiser_best", sr=22050)

        wavelet_denoised_spec = preprocess.wavelet_denoising(data)
        wavelet_denoised_spec_test = preprocess.wavelet_denoising(data_test)
        npt.assert_almost_equal(
                    wavelet_denoised_spec, wavelet_denoised_spec_test,
                    decimal=1)


if __name__ == '__main__':
    unittest.main()
