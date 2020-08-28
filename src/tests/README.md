# Tests

This directory contains the tests that user could perform
Please install the requirements file before running the test cases.

Inorder to run these tests please make sure you follow a similar order, so that the model is created and you could use that model for prediction and detection of 
uncertain samples. Morever, you can also follow different order once you create the model. But make sure that you create the model first before using it!

The command for testing the model_build_and_training script

```
python test_model_build_and_training.py

```
Once the model has been created, you could use the model to generate the report on the test data about its performance.

```

python test_report.py

```
You could also check the test for classifying the calls

```
python test_model_predict.py

```
For testing the uncertain calls script run

```
python test_active_learning.py
```

For testing the preprocessing script run

```
python test_preprocess.py

```
