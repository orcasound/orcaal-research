# Case-1: Preprocessing without PCEN and Wavelet-Denoising

This directory contains following steps that I took when creating colab notebook for preprocessing and trainig.
   
The data used for traning is the audio sound or calls made by Southern Resident Killer Whales(SRKW). This [SRKW data](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive) was provided by [Orcasound organisation](https://www.orcasound.net/).
The data is divided into two parts
  - Audio files
  - .tsv files

## Flowchart

<p align = "center">
<img src = /assets/noPre.png>
</p>

The audio files contains the audio clips consisting of the calls made by the SRKW's whereas the .tsv files contains various parameters of these audio files like the start-time of the call, the duration for which the call was there, date, location, etc

We are also going to use [Ketos library](https://gitlab.meridian.cs.dal.ca/public_projects/ketos) to generate background sounds from the .tsv files and Ketos requires the .tsv files to be in a specific format before being used. Therefore we are going to add an extra parameter to our .tsv files named label which states the labels of the calls. Here it would be SRKW's.

 1. Since the .tsv files in the orcadata repo were not in the format that Ketos library accepted as stated and some samples had start time and an end time equal to zero, you can slightly modified the .tsv's and upload them.
 2. Calculated end-time for the calls and standardized all the .tsv files using Ketos. The reason for standardizing the .tsv is to generate the background sound which we would see later in this section.
 3. Stored all the file_names and their start time in an array which would be used to generate the new audio samples.
 4. Generated new audio files that contain the calls by taking the start time and adding three seconds such that all these new calls(i.e .wav files) have (approximately)only the sound that contains the calls and then saved those files.
 5. The process from step 3 and 4 is repeated for both train and test data and now the .pngs of these calls are stored such that the train folder and test folders contain calls and no calls folders which have the spectrograms of the SRKW calls and background noise respectively.
 6. The spectrograms of the no calls section are generated using Ketos random background generator which extracts only the areas from the .tsv that do not contain the calls by looking at the start time and duration.
 7. Once we have these background sounds of three seconds each, we would convert them to spectrograms and save them.
 8. Now, since we have the spectrograms in both folders, we would create and fine-tune different models for training.
