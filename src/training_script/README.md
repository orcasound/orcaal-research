The directory contains necessary commands for model building and training, predicting and plotting the ROC curve.
model_build_and_training.py: Code for building the model and training
The directory structure should be as follows:
```
Dataset
├── train_srkw
│   ├── calls
│   └── no_calls
└── val_srkw
    ├── calls
    └── no_calls
```
The command to run the training and model building script.
```
python model_build_and_training.py --classpath Path to the training directory --noofepochs No of epochs
```
Please pass the Dataset directory as a training path which would contain the directory structure as shown above.


model_predict.py: This script is used to predict the calls if present.
```
CLASSPATH
├── test_srkw
    ├── calls
    └── no_calls
```
The command to run the predict script is:
```
python model_predict.py --modelpath Path to the model --testpath Path to the test directory that consists of spectrograms to be predicted
```

statistics.py : Would plot the ROC curve

The command to plot the ROC curve is:
```
python statistics.py --modelpath Path to the model --testpath Path to the test directory
```
Make sure to pass the test folders having following directory structure. Here calls and no_calls directories contains the spectrograms of both calls and no_calls respectively

```
CLASSPATH
├── test_srkw
    ├── calls
    └── no_calls
```
The ROC curve similar to this would be plotted
<p align = "center">
<img src = /assets/CNN_final_vs_random.png>
</p>

The command to generate the results of the model using report.py

```
python report.py --modelpath Path to the model --testpath Path to the test directory
```

The command to build docker image
```
sudo docker build -t 'results' .
```
The command to run dockerfile
```
sudo docker run results -m preprocess_mag_scipy_Srkws.h5 -c tests/ 
```

1. classpath: The path to the spectrogram images
2. noofepochs: The number of epochs for which the model is to be trained

- Thanks [Diego](https://github.com/jd-rs) for testing these scripts out on your computer!
- Thanks [OrcaCNN](https://github.com/axiom-data-science/OrcaCNN) as your repository was extremely useful and it helped me alot
