# In this part the preprocessing and training is done using Ketos and the model used for training is simple RNN

## Pipeline

<p align = "center">
<img src = /assets/preprocess2.png>
</p>

The eight stages in the above pipeline are explained below

1. Download and extract the files from the website or AWS server in my case.
2. Add the necessary parameters to the .tsv/.csv files so that the tsv files are in the format accepted by Ketos.
3. Check if the .tsv files are in the standardized format.
4. If they are not in  the standardized format then standardize them using Ketos standardize library.
5. Generate the negative training data by creating a tsv that contains all the duration that are notn within the start time and end time of the positive calls.
6. Append these negative data with the positive ones
7. Define the attributes that are to be used by the spectrograms such as the sampling_rate, maximum and minimum frequency, etc\
8. Convert these audio files into spectrogram data and save them in .hdf5 format in database
