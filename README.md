# AL-MLresearch
This repository contains code, research work, and pre-processing steps involved in developing the Orcacall detection model for the "OrcaAL" Active learning tool during the 2020 Google Summer of Code. The deployment of the model within the tool is described in [the OrcaAL repo](https://orcasound.github.io/orcaal/).

Most of the orca audio data present in the world are unlabeled. Even though labeling data is an expensive, difficult, and slow process, it is an essential part of the advancing machine learning (ML) systems. What if a model could achieve accuracy by annotating a small portion of a large dataset? One way of doing this is [Active Learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) which can allow a human to spend 10-20% of the time annotating data and still get the good model performance.

Therefore, we going to build an active learning tool that would label the vast amount of unlabeled data coming in real-time streams from ocean observing systems.
For more extensive research, jump to the [notebooks directory](https://github.com/orcasound/orcaal-research/tree/master/notebooks).

## Dataset 

This github repository consists of the necessary steps taken for the backend of OrcaAL, i.e. preprocessing, building CNN models, and active learning. Along with this, this repository also quantifies the different accuracies achieved by each model experiment along with the steps necessary steps taken to achieve that accuracy.

I have used the [podcast round 2](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#OrcasoundLab07052019_PodCastRound2) and [podcast round 3](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#OrcasoundLab09272017_PodCastRound3) for training the model. The dataset was neatly labeled by the Orcasound organization using the [Podcast tool](https://github.com/orcasound/orcalabel-podcast) (developed by Microsoft employees through hackathons) which helped me a lot in preprocessing it.

## Implementation

The implementation part consists of the stages:
- Data extraction and preprocessing 
- Model building and training
- Active learning

## General project flow 


<p align = "center">
<img src = 
     /assets/general_stage.png>
</p>

The directory structure of our project looks like this: 

```
├── experiments
│   ├── experiments/Grayscale_magspectrogram_scipy.ipynb <- CNN model trained on grayscale power_spectral_density spectrograms
│   ├── experiments/Grayscale_specs_mag.ipynb  <- Differenet CNN model trained on grayscale power_spectral_density spectrograms
│   ├── experiments/Preprocess_grayscale.ipynb <- Preprocessing of the grayscale spectrograms
│   ├── experiments/README.md
│   ├── experiments/Sir_Val's_method_color.ipynb <- Preprocessing of the power_spectral_density spectrograms as suggested by Val
│   └── experiments/Spectrograms.ipynb <- Different spectrograms that are generated and their code
├── notebooks
│   ├── notebooks/active_learning
│   │   └── notebooks/active_learning/active_learning_pipeline.ipynb <- Active learning pipelne notebook
│   ├── notebooks/case_one
│   │   ├── notebooks/case_one/Preprocessing 
│   │   │   └── notebooks/case_one/Preprocessing/preprocess_case_one.ipynb <- Jupyter notebook in colab performing preprocessing
│   │   ├── notebooks/case_one/README.md
│   │   └── notebooks/case_one/Training
│   │       ├── notebooks/case_one/Training/README.md
│   │       └── notebooks/case_one/Training/training_on_case_one_on_model_VGG_and_CNN.ipynb <- Jupyter notebook in colab performing training.
│   ├── notebooks/case_three
│   │   ├── notebooks/case_three/Preprocess
│   │   │   ├── notebooks/case_three/Preprocess/preprocessing_PCEN_and_Wavlet_denoising.ipynb <- Jupyter notebook in colab performing preprocessing process.
│   │   │   └── notebooks/case_three/Preprocess/README.md
│   │   ├── notebooks/case_three/README.md
│   │   └── notebooks/case_three/Training
│   │       ├── notebooks/case_three/Training/CNN_for_SRKW.ipynb <- Jupyter notebook in colab performing training stage on case three preprocessed spectrograms.
│   │       ├── notebooks/case_three/Training/README.md
│   │       ├── notebooks/case_three/Training/Resnet152_pre_pcen_wd_train.ipynb <- Jupyter notebook in colab performing training stage using Resnet152
│   │       └── notebooks/case_three/Training/Training.ipynb <- Jupyter notebook in colab performing training stage using VGGG-16
│   ├── notebooks/case_two
│   │   ├── notebooks/case_two/Preprocess
│   │   │   └── notebooks/case_two/Preprocess/preprocessing_using_PCEN.ipynb <- Jupyter notebook in colab performing preprocessing process using case two
│   │   └── notebooks/case_two/Training
│   │       └── notebooks/case_two/Training/training_on_PCEN_spectrograms.ipynb <- Jupyter notebook in colab performing training on stage two spectrograms.
│   ├── notebooks/preprocessing_and_training_using_Ketos
│   │   ├── notebooks/preprocessing_and_training_using_Ketos/Preprocessing
│   │   │   ├── notebooks/preprocessing_and_training_using_Ketos/Preprocessing/preprocessing_using_Ketos.ipynb 
│   │   │   └── notebooks/preprocessing_and_training_using_Ketos/Preprocessing/README.md
│   │   └── notebooks/preprocessing_and_training_using_Ketos/Training
│   │       ├── notebooks/preprocessing_and_training_using_Ketos/Training/training_on_RNN_using_Ketos.ipynb
│   │       └── notebooks/preprocessing_and_training_using_Ketos/Training/Training_SRKWs_Ketos.ipynb
│   │    
│   └── notebooks/README.md
├── requirements.txt
├── src
    ├── active_learning_script
│   │   ├── active_learning.py
│   │   └── README.md
│   ├── datasets
│   │   ├── 1562337136_0004.wav
│   │   ├── extracted_calls1.wav
│   │   ├── test_extract_audio.tsv
│   │   ├── test_srkw
│   │   │   ├── calls
│   │   │   │   ├── extracted_calls4.png
│   │   │   │   ├── extracted_calls5.png
│   │   │   │   └── extracted_calls6.png
│   │   │   └── no_calls
│   │   │       ├── extracted_calls1.png
│   │   │       ├── extracted_calls2.png
│   │   │       ├── extracted_calls3.png
│   │   │       ├── extracted_calls4.png
│   │   │       ├── extracted_calls5.png
│   │   │       └── extracted_calls6.png
│   │   ├── test_wav.wav
│   ├── src/preprocessing_script
│   │   ├── src/preprocessing_script/Dockerfile   <- The Dockerfile of the preprocessing script. 
│   │   ├── src/preprocessing_script/preprocess.py <- The preprocessing script. 
│   │   ├── src/preprocessing_script/README.md  
│   │   ├── src/preprocessing_script/requirements.txt <- Requirement file
│   │   └── src/preprocessing_script/selection_table.py <- Selection table for generating background noise
|   ├── tests
│   │   ├── README.md
│   │   ├── test_active_learning.py
│   │   ├── test_model_build_and_training.py
│   │   ├── test_model_predict.py
│   │   ├── test_preprocess.py
│   |   └── test_report.py
|   └── src/training_script
│       ├── src/training_script/Dockerfile <- The Dockerfile of the training script. 
│       ├── src/training_script/model_build_and_training.py <- The model building and training script.
│       ├── src/training_script/README.md
│       └── src/training_script/requirements.txt <- Requirement file
│   
│   
|
│
│   
│
└── trained_models
   └── trained_models/README.md <- Links to the different models that could be used


```


## Contributing to AL-MLresearch
<!--- If your README is long or you have some specific process or steps you want contributors to follow, consider creating a separate CONTRIBUTING.md file--->
To contribute to AL-MLresearch, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <AL-MLresearch>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

Thanks to the following people who have contributed to this project without whom the project would never have been completed.

My Mentors:
 * [@jesselopez](https://github.com/yosoyjay)
 * [@valentina-s](https://github.com/valentina-s) 
 * [@scottveirs](https://github.com/scottveirs) 
 * [@abhisheksingh](https://github.com/ZER-0-NE)
 * [@valveirs](https://github.com/veirs)

My partner who is working on the front-end part
 * [@jorgediego](https://github.com/jd-rs)

Thanks to [OrcaCNN](https://github.com/axiom-data-science/OrcaCNN) and [Ketos](https://gitlab.meridian.cs.dal.ca/public_projects/ketos) for their thorough documentation and code which helped not only me develop preprocessing and training stage but also helped me develop scripts.

## Blog-posts
Here are the links to the three blog-posts that I wrote during gsoc period.
 1. [Detection of Orca calls using Active Learning: GSoC progress report #1](http://www.orcasound.net/2020/07/03/detection-of-orca-calls-using-active-learning/)
 2. [Detection of Orca calls using Active Learning: GSoC progress report #2](http://www.orcasound.net/2020/08/02/detection-of-orca-calls-using-active-learning-gsoc-progress-report-2/)
 3. [Detection of Orca calls using Active Learning: GSoC progress report #3](http://www.orcasound.net/2020/08/26/detection-of-orca-calls-using-active-learning-gsoc-progress-report-3/)
 
## License
<!--- If you're not sure which open license to use see https://choosealicense.com/--->

This project uses the following license: [MIT](https://choosealicense.com/licenses/mit/#).
