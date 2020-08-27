#!python
import argparse
import os
from pathlib import Path

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from pydub import AudioSegment
from scipy import signal
from skimage.restoration import denoise_wavelet

import selection_table



def generate_negative_tsv(
        call_annotations,
        call_time,
        files_dir
):

    """Generates .tsv file containing start-time and end-time of
    the negative calls.

    Since we also want the pure negative samples, that do not contain
    the calls we would generate a .tsv file which contains the interval
    not in the start-time and duration of the .tsv containing the calls.
    And since any area that does not contain the calls would contain no
    call or the background noise, we would use this start-time and
    duration to extract audio from the audio files.

    Args:
        call_annotations: The .tsv file containing the calls.
        call_time: The duration for which you want to generate negative
                    calls.
        files_dir: The directory that contains the audio data.

    Returns:
        A pandas dataframe containing start_time and end_time of the
        background sounds.
    """
    standardized_annotations = selection_table.standardize(
        table=call_annotations,
        signal_labels=["SRKWs"],
        mapper={"wav_filename": "filename"},
        trim_table=True
    )

    positives_call_duration = selection_table.select(
        annotations=standardized_annotations,
        length=call_time
    )
    file_durations = selection_table.file_duration_table(
        files_dir
    )

    # Generate a .tsv file which does not include any calls.
    negatives_annotations = selection_table.create_rndm_backgr_selections(
        annotations=standardized_annotations,
        files=file_durations,
        length=call_time,
        num=len(positives_call_duration),
        trim_table=True
    )

    negative_tsv_generated = negatives_annotations.reset_index(level=[0, 1])

    return negative_tsv_generated


def extract_audio(
        output_directory,
        file_location,
        call_time_in_seconds,
        call_annotations
):
    """This function extracts the audio of a specified duration.

    Since a single audio clip might consist of a mixture of both calls
    and no calls, therefore smaller audio clips of particular time frame
    are extracted to get the complete positive and negative calls. These
    calls are extracted by taking the start-time from the .tsv file and
    the duration of the call as specified by the user.

    Args:
        label: A string specifying whether we are extracting calls or
            no call.
        tsv_filename: The .tsv file containing the parameters like start-time,
            duration, etc.
        output_directory: The path output directory where we want to store
            these extracted calls.
        file_location: The location of the audio file in .wav format.
        call_time_in_seconds: Enter the duration of calls you want
            to extract in seconds.Integer value.

    Returns:
        None
    """

    file_name = call_annotations.filename[:].values
    start_time = call_annotations.start[:].values

    i = 0
    call_duration = 0
    call_time_in_seconds = call_time_in_seconds*1000

    for audio_file in file_name:
        audio_file = os.path.join(file_location, audio_file)
        sound = AudioSegment.from_file(audio_file)
        start_time_duration = start_time[i]
        start_time_duration = start_time_duration * 1000
        i = i + 1
        call_duration = start_time_duration + call_time_in_seconds
        call = sound[start_time_duration:call_duration]
        output_file = os.path.join(
                        output_directory,
                        "extracted_calls{0}.wav".format(i))
        call.export(output_file, format="wav")


def apply_per_channel_energy_norm(spectrogram):
    """Apply PCEN.

    This function normalizes a time-frequency representation S by
    performing automatic gain control, followed by nonlinear compression:

    P[f, t] = (S / (eps + M[f, t])**gain + bias)**power - bias**power
    PCEN is a computationally efficient frontend for robust detection
    and classification of acoustic events in heterogeneous environments.

    This can be used to perform automatic gain control on signals that
    cross or span multiple frequency bans, which may be desirable
    for spectrograms with high frequency resolution.

    Args:
        spectrograms: The data from the audio file used to create spectrograms.
        sampling_rate: The sampling rate of the audio files.

    Returns:
        PCEN applied spectrogram data.
    """

    pcen_spectrogram = librosa.core.pcen(spectrogram)
    return pcen_spectrogram


def wavelet_denoising(spectrogram):
    """In this step, we would apply Wavelet-denoising.

    Wavelet denoising is an effective method for SNR improvement
    in environments with a wide range of noise types competing for the
    same subspace.

    Wavelet denoising relies on the wavelet representation of
    the image. Gaussian noise tends to be represented by small values in the
    wavelet domain and can be removed by setting coefficients below
    a given threshold to zero (hard thresholding) or
    shrinking all coefficients toward zero by a given
    amount (soft thresholding).

    Args:
        data: Spectrogram data in the form of NumPy array.

    Returns:
        Denoised spectrogram data in the form of numpy array.
    """
    im_bayes = denoise_wavelet(
        spectrogram,
        multichannel=False,
        convert2ycbcr=False,
        method="BayesShrink",
        mode="soft"
    )
    return im_bayes


def plot_power_spectral_density(data,
                                samplerate,
                                f_name,
                                plot_path,
                                grayscale=False):
    """Plot power spectral density spectrogram

    Compute a spectrogram with consecutive Fourier transforms.

    Spectrograms can be used as a way of visualizing the
    change of a nonstationary signal’s frequency content over time.

    Args:
        data: Spectrgram data in the form of NumPy array.
        samplerate: Sampling rate
        f_name: The name of the audio file
        plot_path: The path to the directory where we want to plot the
                    spectrogram
        grayscale: The color map of the spectrogram

    Returns:
        None
    """

    f, t, spec = signal.spectrogram(data, samplerate)
    fig, ax = plt.subplots(1, 1)
    if grayscale:
        ax.specgram(data, Fs=samplerate, cmap="gray", NFFT=1024)
    else:
        ax.specgram(data, Fs=samplerate, NFFT=1024)
    scale_y = 1000
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
    ax.yaxis.set_major_formatter(ticks_y)
    ax.set_axis_off()
    plt.savefig(os.path.join(
                plot_path,
                f_name[:-4] + ".png"))
    plt.close(fig)


def spec_plot_and_save(denoised_data, f_name, output_dir):
    """Generate the spectrogram and save them.

    Args:
        denoised_data: The spectrogram data that is generated either by
        PCEN or Wavelet-denoising.
        f_name: The name of the output file.
        output_dir: The path to the output directory.

    Returns:
       None.
    """
    fig, ax = plt.subplots()
    i = 0
    ax.imshow(denoised_data)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.set_size_inches(10, 10)
    fig.savefig(
        os.path.join(output_dir, f"{f_name[:-4]}" + "_{:04d}.png".format(i)), dpi=80, bbox_inches="tight", quality=95, pad_inches=0.0
                    )
    fig.canvas.draw()
    fig.canvas.flush_events()
    i += 1
    plt.close(fig)


def select_spec_case(
                    plot_path,
                    folder_path,
                    melspectrogram=False,
                    pcen=False,
                    wavelet=False,
                    psd=False,
                    grayscale=False):
    """Selects the preprocessing steps to be applied to the spectrogram.

    Depending upon the choices entered by the user this function would
    select the necessary preprocessing stages and call their respective
    functions.

    Args:
        plot_path: The output path where we want to plot the spectrograms.
        folder: The input_path which contains the audio that would
            be used to generate spectrograms.
        pcen: Could be set to True if we want to apply PCEN to spectrograms.
        wavelet: Could be set to true if we want to apply Wavelet denoising
            to the spectrograms.

    Returns:
        None.
    """
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for id, file in enumerate(onlyfiles):
        fpath = os.path.join(folder_path, file)
        data, sr = librosa.core.load(fpath, sr=20000, res_type="kaiser_best")
        f_name = os.path.basename(file)
        spectrogram_data = librosa.feature.melspectrogram(
                                                      data,
                                                      sr=20000,
                                                      power=1)
        if psd:
            plot_power_spectral_density(data, sr, f_name, plot_path, grayscale)

        elif melspectrogram is True and pcen is False and wavelet is False:
            spec_plot_and_save(spectrogram_data, f_name, plot_path)

        elif (melspectrogram is True and pcen is True and wavelet is False):
            pcen_spec = apply_per_channel_energy_norm(spectrogram_data)
            spec_plot_and_save(pcen_spec, f_name, plot_path)

        elif (melspectrogram is True and pcen is True and wavelet is True):
            pcen_spec = apply_per_channel_energy_norm(spectrogram_data)
            spectrogram_data = wavelet_denoising(pcen_spec)
            spec_plot_and_save(spectrogram_data, f_name, plot_path)


def main(
        tsv_path,
        files_dir,
        call_time,
        output_dir,
        melspectrogram,
        pcen,
        wavelet,
        power_spectral_density,
        grayscale):
    # prepare output directories
    positive_dir = os.path.join(output_dir, "positive_calls")
    if not os.path.isdir(positive_dir):
        os.mkdir(positive_dir)

    negative_dir = os.path.join(output_dir, "negative_calls")
    if not os.path.isdir(negative_dir):
        os.mkdir(negative_dir)

    positive_plot_dir = os.path.join(output_dir, "positive_plots")
    if not os.path.isdir(positive_plot_dir):
        os.mkdir(positive_plot_dir)

    negative_plot_dir = os.path.join(output_dir, "negative_plots")
    if not os.path.isdir(negative_plot_dir):
        os.mkdir(negative_plot_dir)

    # load tsv file
    call_annotations = pd.read_csv(tsv_path, sep="\t")
    try:
        call_length_mean = call_annotations["duration_s"].mean()
        print("The mean of the call duration is {}".format(call_length_mean))
    except Exception:
        print("Please change the call duration label in your .tsv file by 'duration_s' ")
    try:
        call_annotations["end"] = call_annotations["start"] + call_annotations["duration_s"]
    except Exception:
        print("Please change the start time of the call label in your .tsv to start")

    # extract the audio of the calls
    extract_audio(
        positive_dir,
        files_dir,
        call_time,
        call_annotations
    )

    # generate negative .tsv file
    negative_generated_tsv = generate_negative_tsv(
                                    call_annotations,
                                    call_time, files_dir)

    # extract the audio of the negative calls or background calls

    extract_audio(
        negative_dir,
        files_dir,
        call_time,
        negative_generated_tsv
    )

    # select the spectrogram that you want to plot
    select_spec_case(
        positive_plot_dir,
        positive_dir,
        melspectrogram,
        pcen,
        wavelet,
        power_spectral_density,
        grayscale
    )

    select_spec_case(
        negative_plot_dir,
        negative_dir,
        melspectrogram,
        pcen,
        wavelet,
        power_spectral_density,
        grayscale
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess audio files for use with CNN models"
    )
    parser.add_argument(
        "--tsv_path",
        type=str,
        help="Path to tsv file",
    )

    parser.add_argument(
        "--files_dir",
        type=str,
        help="Path to directory with audio files"
    )
    parser.add_argument(
        "--call_time",
        type=int,
        help="Target length of processed audio file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output directory"
    )
    parser.add_argument(
        "--melspectrogram",
        action="store_true",
        help="Plot melspectrogram"
    )
    parser.add_argument(
        "--pcen",
        action="store_true",
        help="Apply PCEN"
    )
    parser.add_argument(
        "--wavelet",
        action="store_true",
        help="Apply wavelet denoising"
    )
    parser.add_argument(
        "--power_spectral_density",
        action="store_true",
        help="Plot power spectral density spectrogram"
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Plot the grayscale spectrogram"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main(
        args.tsv_path,
        args.files_dir,
        args.call_time,
        args.output_dir,
        args.melspectrogram,
        args.pcen,
        args.wavelet,
        args.power_spectral_density,
        args.grayscale
    )
