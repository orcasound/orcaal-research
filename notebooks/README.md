This directory contains the different jupyter notebooks on colab which describes the different preprocessing steps taken in detail along with different models used and the different accuracy that was achieved by each model. Moreover, this directory also contains the effect of Active learning and how the performance of the model improved with the help of active learning.
The three cases are explained below in brief:

The three cases contains three different ways in which data is preprocessed and passed to different models for performing predictions.
Below is the flowchart that explains each of the case and the models used for training in each cases

# General flow 


<p align = "center">
<img src = 
     /assets/General_flow.png>
</p>

## The three cases are:
 - Preprocessing without applying PCEN and Wavelet-Denoising
 - Preprocessing using PCEN
 - Preprocessing using PCEN and Wavelet-Denoising

### The different models that are used in the above cases are:
1. Preprocessing without PCEN and Wavelet Denosing  
    1. Basic Convolution Neural Network
    2. VGG-16
  
2. Preprocessing with PCEN 
    1. VGG-16
  
3. Preprocessing with PCEN and Wavelet Denoising
    1. Resnet-512
    2. VGG-16
    3. Basic CNN 
    4. InceptionResnet-V2

#### Note: Although, in the directory structure we have preprocessing and training using Ketos we are not going to use this phase for active learning, but we are going to use the three cases for the active learning.
# Active Learning Phase
This is the active learning phase that would be used to evaluate the outcome of the active learning on the model where a small subset would be extracted and the model would perform probability predictions on these subset, which depending on the probability would be passed to the labeler or directly to the model.


# Active Learning flow 1


<p align = "center">
<img src = /assets/active_final.png>
</p>

I have worked on this active learning pipeline where I have taken the following steps:

1. Preprocess the spectrograms.
2. Create a CNN model and train the training data(Podcast Round 2 and Round 3) on the CNN model.
3. Calculate the Probability predictions for each of the samples present in the test data.
4. If the model predicts the probability of the sample (being either call or no-call) in the range of 0.4 to 0.6, the model is uncertain and pass
    these calls to the labeler to label them.
5. But, if the value of the predicted sample is greater than 0.6 assign it no call, and if the value is less then 0.4 assign the sample call.
6. Retrain the model on the combined data(samples labeled by the model as well as the user).

# Active Learning flow 2
The idea of choosing the threshold between 0.1 to 0.9 for uncertainty range was adviced by my mentor [Jesse](https://github.com/yosoyjay) and an increase in accuracy was found because of this method as compared to the previous method.

<p align = "center">
<img src = /assets/ssf.png>
</p>

The steps taken in the above flowchart are as follows:
1. Preprocess Spectrograms: Generate Melspectrograms with the help of and librosa library  and then apply PCEN and Wavelet-denoising. These spectrograms are generated from the audio files containing calls and no calls.
2. Train our CNN model on training data.
Note: A small subset from the training data has been removed for active learning on which the model is not trained on.
3. Test the accuracy of the model on the test data.
4. Use this model to perform probability prediction on the subset of the training sample and check if the probability prediction is between 0.1 and 0.9.
5. If yes, ask experts like Scott, Val to label them and pass them to the training directory with True labels.
6. If no, then pass ask for the next batch of samples to be labeled.
7. Retrain the model with this new data along with the old data.
8. Check the accuracy of the model on test data.
The distribution of the training, active learning, retraining, and test dataset is the same as the previous blog. Here, is the distribution chart:

There were 163 uncertain calls that the model detected. There are 12 confident calls and 1 confident no call.
These 163 uncertain samples are being labeled and then sent again with the training dataset to perform training the model. The new accuracy of the model was found to be 84%.


## The ROC curve

<p align = "center">
<img src = /assets/CNN_final_vs_random.png>
</p>

The CNN model of the case three generates the following ROC curve.
