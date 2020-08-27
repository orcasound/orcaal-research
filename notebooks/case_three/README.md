# Case-3: Preprocessing using both PCEN and Wavelet-denoising and then traning the data

This is the case where both the preprocessing stages PCEN and Wavelet-denoising have been applied to improve the quality of the spectrograms that we would use for traning on different models.

# Flowchart


<p align = "center">
<img src = /assets/pre_train.png>
</p>



The audio files contains the audio clips consisting of the calls made by the SRKW's whereas the .tsv files contains various parameters of these audio files like the start-time of the call, the duration for which the call was there, date, location, etc

We are also going to use [Ketos library](https://gitlab.meridian.cs.dal.ca/public_projects/ketos) to generate background sounds from the .tsv files and Ketos requires the .tsv files to be in a specific format before being used. Therefore we are going to add an extra parameter to our .tsv files named label which states the labels of the calls. Here it would be SRKW's.

1. Since the .tsv files in the orcadata repo were not in the format that Ketos library accepted and some samples had start time and an end time equal to zero, I slightly modified the .tsv's and uploaded from my local pc.
2. Calculated end-time for the calls and standardized all the .tsv files using Ketos. The reason for standardizing the .tsv is to generate the background sound which we would see later in this section.
3. Stored all the file_names and their start time in an array which would be used to generate the new audio samples.
4. Generated new audio files that contain the calls by taking the start time and adding three seconds such that all these new calls(i.e .wav files) have (approximately)only the sound that contains the calls and then saved those files.
5. From these newly generated files, I first applied PCEN and then wavelet denoising, and then I plotted the spectrograms using the denoised data generated from the wavelet denoising stage and saved them.
6. The process from step 3 to step 5 is repeated for both train and test data and now the .pngs of these calls are stored such that the train folder and test folders contain calls and no calls folders which have the spectrograms of the SRKW calls and background noise respectively.
7. The spectrograms of the no calls section are generated using Ketos random background generator which extracts only the areas from the .tsv that do not contain the calls by looking at the start time and duration.
8. Once we have these background sounds of three seconds each, we would apply pcen and wavelet denoising to them as well and save them.
9. Now, since we have the spectrograms in both folders, I finetuned VGG-16 model for training.

MODEL TRAINING PHASE
## VGG-16

1. Downloaded the VGG16 model.
2. Removed the last layer from vgg16 and added dense layer for the prediction of two classes(SRKW calls or no calls).
3. Trained the data for 55 epochs on VGG16 and it gave an accuracy of 99% and the loss around 0.022(it was overfitting) but it predicted well, I have also plotted the actual labels and the predicted labels by the model.

## ResNet-512

1. Downloaded the ResNet-512 model.
2. Removed the last layer from ResNet-512 and added dense layer for the prediction of two classes(SRKW calls or no calls).

##  CNN model
 
1. Build CNN model from scratch.


