This script would determine the uncertain samples that the model predicts.Currently the uncertainity range is between the range of 0.1 to 0.9.
The samples classified in this range would be placed in a seperate folder, which experts like [Scott](https://github.com/scottveirs) and [Val](https://github.com/veirs) could label.

The command to run the script is:
```
python3 active_learning.py -c Path to the spectrograms images -m path to the model

```
An example to run this script is 

```
python3 active_learning.py -c test/calls/ -m modelpreprocess_mag_scipy_Srkws.h5
```
 -  -c: Path to the spectrograms
 - -m: Path to the model

Note: Please make sure to select the same model for the same type of spectrogram. If you are using different model make sure to change the img_width, img_height
