### Preprocessing audio files
The preprocessing script is used convert the audio files into spectrograms containing calls and no calls. The spectrogram type is specified by the user
The script takes audio files and tsv file specifying the parameters like the filename, start, duration_s, label as input and extracts audio based on the start_time and the duration specified by the user. With the help of tsv file given by the user another tsv file is generated that contains the start_time and duration of the background noise(i.e the area in which no call is present).
With the help of these two .tsv files the audio is extracted and spectrograms specified by the user are generated based on these shorter audio files.

### Note: Selection table was created by [Ketos](https://docs.meridian.cs.dal.ca/ketos/introduction.html) library!
Please note that the following headers are required for the preprocessing script filename, start, duration_s and label. If you have some other header please change the .tsv with the following ones.

| filename | start	| duration_s |	label |
  | --------------------------- | ---------------- | --------------------- | ------------------------- |

- filename: name of the audio file
- start: start time of the call
- duration_s: duration of the call in seconds
- label: label to which the call belongs

To run the preprocessing script use the following command
```
python preprocess.py --tsv_path PATH TO THE TSV  --files_dir PATH TO THE AUDIO FILES --call_time DURATION OF THE CALLS --output_dir PATH TO THE OUTPUT DIRECTORY --power_spectral_density --grayscale 
```

### Here, is an example using Docker

### Docker command to build(here preprocess is the name, I have given, you could give whatever you want)
```
sudo docker build -t 'preprocess' .
```
### Docker command to run the process(you could run any of the preprocessing command)

```
sudo docker run -ti preprocess --tsv_path podcast2.tsv --files_dir Round2_OS_07_05/wav/  --call_time 3 --output_dir output --power_spectral_density --grayscale
```

### If you want to explore the container

```
 sudo docker ps -a
```

### After finding the name of the container run the following command, here stupefied_dirac is my container name, replace it with your container name

```
sudo docker start stupefied_dirac
```

### Copy those commands to the output directory and replace the 'output_path' by your output directory name where you want your output

```
 sudo docker cp stupefied_dirac:/usr/src/app/output/ output_path/
```


These are the different types of spectrograms that the user could plot with the help of given commands!
<table border="0">
 <tr>
    <td><b style="font-size:30px">Spectrogram command</b></td>
  <td><b style="font-size:30px">Spectrogram generated</b></td>
 </tr>
 
 <tr>
    <td>--power_spectral_density</td>
     <td><p align = "center">
<img src = /assets/psd_color_scipy.png>
</p>
</td>
 </tr>
  <tr>
    <td>--power_spectral_density --grayscale</td>
     <td><p align = "center">
<img src = /assets/grayscale_psd.png>
</p>
</td>
 </tr>
  <tr>
    <td>--melpectrogram</td>
     <td><p align = "right">
<img src = /assets/melscale.png>
</p>
</td>
 </tr>
    
   <tr>
    <td>--melspectrogram --pcen</td>
     <td><p align = "right">
<img src = /assets/pcen_melspectrogram.png>
</p>
</td>
 </tr>
    
   <tr>
    <td>--melspectrogram --pcen --wavelet</td>
     <td><p align = "right">
<img src = /assets/wavelet_denoising_mel.png>
</p>
</td>
 </tr>
 
</table>

