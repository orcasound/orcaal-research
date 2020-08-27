This directory contains various experiments performed with different types of spectrograms on different models, and an accuracy chart giving the accuracy of different types of spectrograms on the CNN model.
Please take a look at [these sheets](https://docs.google.com/spreadsheets/d/1YcYTjDfneUiaT3ZFwKXkP3U6pY4OjneXEDFzNwZyM2s/edit?usp=sharing) for more information about the model and parameters


- ### Accuracy chart of different spectrograms
    | Type of spectrograms | Color-colormap | Accuracy on test-set |  Accuracy on training-set |                                                           
    | --------------------------- | ---------------- | --------------------- | ------------------------- |
    |  Magnitude Spectrogram | Grayscale-Greys  | 75.70%                | 75.29%                    |
    |  Magnitude Spectrogram    | RGB-viridis	     | 79%                   | 95%                       |
    |  Mel-Spectrograms PCEN and Wavelet Denoising  | RGB-viridis      | 81%                   | 88%-91%                   |
    |  Mel-Spectrograms | Grayscale-Greys | 75% |  |


## Here are different spectrograms generated on one of the .wav file that contains the calls

<table border="0">
 <tr>
    <td><b style="font-size:30px">Title</b></td>
 </tr>
 <tr>
    <td>PSD_Spectrogram</td>
     <td><p align = "center">
<img src = /assets/psd_color_scipy.png>
</p>
</td>
 </tr>
  <tr>
    <td>PSD_Grayscale_Spectrogram</td>
     <td><p align = "center">
<img src = /assets/grayscale_psd.png>
</p>
</td>
 </tr>
  <tr>
    <td>Mel_Spectrogram</td>
     <td><p align = "right">
<img src = /assets/melscale.png>
</p>
</td>
 </tr>
    
   <tr>
    <td>Mel_Spectrogram after PCEN</td>
     <td><p align = "right">
<img src = /assets/pcen_melspectrogram.png>
</p>
</td>
 </tr>
    
   <tr>
    <td>Mel_Spectrogram after PCEN and Wavelet Denoising</td>
     <td><p align = "right">
<img src = /assets/wavelet_denoising_mel.png>
</p>
</td>
 </tr>
    
     
   <tr>
    <td>Grayscale Mel_Spectrogram after PCEN and Wavelet Denoising</td>
     <td><p align = "right">
<img src = /assets/greyscale(2).png>
</p>
</td>
 </tr>
    
          
          
 
 
</table>
