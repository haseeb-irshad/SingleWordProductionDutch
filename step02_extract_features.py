import os
import numpy as np
import scipy.signal
import scipy.io.wavfile
import scipy.fftpack
import torch
from speechbrain.lobes.models.FastSpeech2 import mel_spectrogram
from step01_data_loader import BIDSDataLoader

# compute the Hilbert transform at the next fast FFT length because FFTs are much slower for lengths with large prime factors
# scipy.signal.hilbert zero-pads the signal to that length and the slice trims the output back to the original number of samples
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)), axis = 0)[:len(x)]

def get_num_windows(n_samples, sr, windowLength, frameshift):
    """
    Calculate the total number of sliding time windows for a given signal length

    Parameters
    ----------
    n_samples : int
        Total number of samples in the time series
    sr : float or int
        Sampling rate of the signal in Hz
    windowLength : float
        Length of window (in seconds) over which features are extracted
    frameshift : float
        Shift (in seconds) after which the next window will be extracted

    Returns
    -------
    numWindows : int
        Total number of complete sliding windows that fit within the sample length
    """
    # convert total sample count to total signal duration in seconds
    total_duration = n_samples / sr

    # return 0 if the total signal is shorter than a single window length
    if total_duration < windowLength:
        return 0

    # calculate remaining time after initial window
    # floor to drop incomplete trailing frames, add 1 for initial window and cast to int for array indexing
    return int(np.floor((total_duration - windowLength) / frameshift)) + 1

def extractHG(data, sr, windowLength, frameshift):
    """
    Extract High-Gamma (70-170 Hz) amplitude envelope features from raw iEEG data

    Applies common average referencing (CAR), linear detrending, High-Gamma bandpass
    filtering, mains noise notch filtering (100 Hz and 150 Hz), analytic envelope extraction
    via Hilbert transform, logarithmic scaling, windowed averaging and standardisation

    The common average reference is computed as the median across only the 'valid' channels
    (near-zero variance or outlier variance channels excluded)

    de Cheveigné and Arzounian (2018) describe robust detrending as an alternative to linear
    detrending that iteratively down-weights glitch-contaminated samples so they do not bias
    the fit, along with related methods (robust rereferencing, outlier detection, inpainting)

    https://pmc.ncbi.nlm.nih.gov/articles/PMC5915520/

    Parameters
    ----------
    data : np.ndarray
        Multichannel iEEG time series array formatted with shape (samples, channels)
    sr : float or int
        Sampling rate of the iEEG data in Hz
    windowLength : float
        Length of window (in seconds) over which features are averaged
    frameshift : float
        Shift (in seconds) between consecutive sliding windows

    Returns
    -------
    feat : np.ndarray
        Z-scored High-Gamma log-amplitude feature matrix formatted with shape (windows, channels)
    """
    # identify responsive non-dead channels based on standard deviation boundaries
    ch_stds = np.nanstd(data, axis = 0)
    median_std = np.nanmedian(ch_stds)
    valid_mask = (ch_stds > 1e-6) & (ch_stds < 5 * median_std)
    valid_indices = np.where(valid_mask)[0]

    # common average referencing (CAR) using robust median across valid channels
    common_average_reference = np.nanmedian(data[:, valid_indices], axis = 1, keepdims = True) # median reference is robust to outlier channels
    data_car = data - common_average_reference

    # linear detrending across time to remove low-frequency drift
    data_detrend = scipy.signal.detrend(data_car, axis = 0)

    # bandpass filter for High-Gamma (70-170 Hz)
    sos_hg = scipy.signal.iirfilter(4, [70.0 / (sr / 2.0), 170.0 / (sr / 2.0)], btype = 'bandpass', output = 'sos')
    x_bp = scipy.signal.sosfiltfilt(sos_hg, data_detrend, axis = 0)

    # notch filtering for mains power line harmonics (100 Hz and 150 Hz)
    # a 4 Hz bandstop width is used to suppress the narrowband mains peak without attenuating a wider slice of the surrounding High-Gamma band than necessary
    sos_notch1 = scipy.signal.iirfilter(4, [98.0 / (sr / 2.0), 102.0 / (sr / 2.0)], btype = 'bandstop', output = 'sos')
    x_notch1 = scipy.signal.sosfiltfilt(sos_notch1, x_bp, axis = 0)

    sos_notch2 = scipy.signal.iirfilter(4, [148.0 / (sr / 2.0), 152.0 / (sr / 2.0)], btype = 'bandstop', output = 'sos')
    x_notch2 = scipy.signal.sosfiltfilt(sos_notch2, x_notch1, axis = 0)

    # extract analytic envelope via Hilbert transform and apply logarithmic compression
    env = np.abs(hilbert3(x_notch2))
    env = np.log(env + 1e-10)

    samples = env.shape[0]
    numWindows = get_num_windows(samples, sr, windowLength, frameshift)

    feat_local = np.zeros((numWindows, env.shape[1]))
    for window in range(numWindows):
        t_start = window * frameshift
        t_end = t_start + windowLength
        idx_start = int(np.round(t_start * sr))
        idx_end = min(int(np.round(t_end * sr)), samples)

        # mean log-amplitude is used within the window, as averaging over many samples gives a low-variance estimate
        feat_local[window, :] = np.mean(env[idx_start:idx_end, :], axis = 0)
    
    # Z-score feature matrix per channel across time
    feat = (feat_local - np.mean(feat_local, axis = 0)) / (np.std(feat_local, axis = 0) + 1e-10)

    return feat

def stackFeatures(features, modelOrder, stepSize):
    """
    Add temporal context to each window by stacking neighbouring past and future feature vectors

    Parameters
    ----------
    features : np.ndarray
        Feature matrix formatted with shape (windows, channels)
    modelOrder : int
        Number of temporal context frames to include prior to and after the current window
    stepSize : int
        Frame step size used when sampling temporal context frames

    Returns
    -------
    featStacked : np.ndarray
        Temporally stacked feature matrix formatted with shape
        (windows - 2 * modelOrder * stepSize, channels * (2 * modelOrder + 1))
    """
    featStacked = np.zeros((features.shape[0] - (2 * modelOrder * stepSize), (2 * modelOrder + 1) * features.shape[1]))
    for fNum, i in enumerate(range(modelOrder * stepSize, features.shape[0] - modelOrder * stepSize)):
        ef = features[i - modelOrder * stepSize : i + modelOrder * stepSize + 1 : stepSize, :]
        featStacked[fNum, :] = ef.flatten()
    return featStacked

def downsampleLabels(labels, sr, windowLength, frameshift):
    """
    Downsample a categorical label time series by computing the modal value within each sliding time window

    Parameters
    ----------
    labels : np.ndarray
        Categorical or text label time series array
    sr : float or int
        Sampling rate of the label data in Hz
    windowLength : float
        Length of window (in seconds) over which the modal label is determined
    frameshift : float
        Shift (in seconds) between consecutive sliding windows

    Returns
    -------
    newLabels : np.ndarray
        Downsampled array of categorical string labels corresponding to each sliding window
    """
    numWindows = get_num_windows(len(labels), sr, windowLength, frameshift)
    newLabels = np.empty(numWindows, dtype = 'U50')
    for w in range(numWindows):
        t_start = w * frameshift
        t_end = t_start + windowLength

        idx_start = int(np.round(t_start * sr))
        idx_end = min(int(np.round(t_end * sr)), len(labels))

        window = labels[idx_start : idx_end]
        if window.size == 0:
            newLabels[w] = ""
            continue
        uniq, counts = np.unique(window, return_counts = True)
        mode = uniq[np.argmax(counts)]
        newLabels[w] = mode
    return newLabels

def extractMelSpecs(audio, sr):
    """
    Extract log-Mel spectrogram features compatible with SpeechBrain HiFi-GAN vocoder requirements

    The hop_length, win_length, n_mels, n_fft and f_max values below are hardcoded to match
    the specific pretrained vocoder this project uses (speechbrain/tts-hifigan-libritts-16kHz),
    which was trained on 16000 Hz audio with these exact parameters. For this dataset and
    vocoder, sr must therefore be 16000 Hz. To reuse this function with a different dataset or a
    different vocoder, update sr together with the mel_spectrogram parameters below to match that
    vocoder's own training configuration

    Parameters
    ----------
    audio : np.ndarray or torch.Tensor
        Raw 1D audio time series signal
    sr : int
        Sampling rate of the audio in Hz

    Returns
    -------
    spectrogram : np.ndarray
        Logarithmic Slaney-scaled 80-bin Mel spectrogram matrix formatted with shape (windows, 80).
        Frames are start-anchored (frame k spans samples [k * 256, k * 256 + 1024)) so that they align
        with the iEEG windows produced by extractHG
    """
    # convert raw NumPy audio array to PyTorch FloatTensor
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float()
    else:
        audio_tensor = audio.float()

    # removes a leading singleton batch dimension if present (e.g. [1, samples] -> [samples])
    # does not handle true multi-channel audio (e.g. [2, samples] stereo)
    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.squeeze(0)

    # STFT framing parameters matched to the pretrained HiFi-GAN vocoder
    hop_length = 256    # 16 ms shift at 16 kHz
    n_fft = 1024        # 64 ms window at 16 kHz

    # call SpeechBrain's native HiFi-GAN Mel pipeline
    # returns a tuple (mel_spectrogram, energy), so we catch just the mel_spec
    mel_spec, _ = mel_spectrogram(
        audio = audio_tensor,
        sample_rate = sr,
        hop_length = hop_length,
        win_length = n_fft,
        n_mels = 80,
        n_fft = n_fft,
        f_min = 0.0,
        f_max = 8000.0,    # maximum frequency boundary
        power = 1,         # extracting magnitude spectrum (|STFT|) rather than power
        normalized = False,
        min_max_energy_norm = True,
        norm = "slaney",
        mel_scale = "slaney",
        compression = True,
    )

    # transpose [80, T] to [T, 80] so time frames are rows
    # .cpu().numpy() transfers tensor to CPU memory and converts to standard NumPy array
    spectrogram = mel_spec.T.cpu().numpy()

    # torchaudio centres STFT frames by padding n_fft // 2 samples at the start (center = True)
    # drop the leading padded frames so frame k starts at sample k * hop_length, matching extractHG windows
    assert (n_fft // 2) % hop_length == 0, "n_fft / 2 must be a multiple of hop_length for exact alignment"
    offset = (n_fft // 2) // hop_length # 2 frames (32 ms)
    spectrogram = spectrogram[offset:, :]

    return spectrogram

def nameVector(elecs, modelOrder, stepSize):
    """
    Generate electrode feature names by appending temporal context offsets for stacked feature matrices

    Parameters
    ----------
    elecs : np.ndarray or list of str
        Original electrode channel names
    modelOrder : int
        Number of past and future temporal context frames included per window
    stepSize : int
        Frame step size used when sampling temporal context frames

    Returns
    -------
    names : np.ndarray
        1D array of feature names incorporating temporal offset suffixes (e.g. 'LA1T-12', 'LA1T0', 'LA1T+12')
    """
    # standardise input to a clean 1D array of channel string names
    elec_names = np.asarray(elecs).flatten()
    feature_names = []

    # outer loop: iterate through each time offset
    for off in range(-modelOrder, modelOrder + 1):
        frame_offset = off * stepSize
        offset_str = f"T{frame_offset}" if frame_offset <= 0 else f"T+{frame_offset}"
        # inner loop: iterate through all electrode channels for the current time offset
        for name in elec_names:
            feature_names.append(f"{name}{offset_str}")
    
    return np.array(feature_names)

if __name__ == "__main__":
    windowLength = 0.064 # 64 ms window duration
    frameshift = 0.016   # 16 ms window frameshift
    modelOrder = 4       # 4 windows past/future temporal context
    stepSize = 3         # 48 ms context steps (3 frames * 16 ms shift)

    # determine script directory and build absolute BIDS path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_bids = os.path.join(script_dir, 'SingleWordProductionDutch-iBIDS')
    path_output = os.path.join(script_dir, 'features')
    os.makedirs(path_output, exist_ok = True)

    # initialise BIDSDataLoader
    loader = BIDSDataLoader(bids_root = path_bids)
    participants_df = loader.load_participants_table()

    for participant in participants_df['participant_id']:
        print(f"Extracting features for participant: {participant}")

        # load raw data container via BIDSDataLoader
        raw_data = loader.load_raw_participant(participant)

        eeg, eeg_sr = raw_data.eeg, raw_data.eeg_sr
        audio, audio_sr = raw_data.audio, raw_data.audio_sr
        words = raw_data.words
        channels = raw_data.channels['name'].values

        # extract HG features
        feat = extractHG(eeg, eeg_sr, windowLength, frameshift)

        # process audio
        target_sr = 16000
        audio = scipy.signal.decimate(audio, int(audio_sr / target_sr)) # scipy.signal.resample_poly can be used instead of decimate for non-integer sampling rates

        # synchronise audio length to exact duration of iEEG signal
        max_audio_samples = int(np.round((eeg.shape[0] / float(eeg_sr)) * target_sr))
        audio = audio[:max_audio_samples]

        # prepare normalised float array [-1.0, 1.0] for PyTorch/SpeechBrain STFT
        audio_norm = audio / np.max(np.abs(audio))

        # write 16-bit PCM WAV file for playback/reference
        scaled = np.int16(audio_norm * 32767)
        scipy.io.wavfile.write(os.path.join(path_output, f'{participant}_orig_audio.wav'), target_sr, scaled)

        # extract spectrogram
        melSpec = extractMelSpecs(audio_norm, target_sr)

        # synchronise frame counts between EEG features and audio spectrogram
        min_frames = min(feat.shape[0], melSpec.shape[0])
        feat = feat[:min_frames, :]
        melSpec = melSpec[:min_frames, :]

        # stack features
        feat_stacked = stackFeatures(feat, modelOrder, stepSize)

        # Z-score log-Mel spectrogram across time per frequency bin
        # save mean and std so that step04 can undo the Z-scoring
        mel_mean = np.mean(melSpec, axis = 0)
        mel_std = np.std(melSpec, axis = 0)
        melSpec = (melSpec - mel_mean) / (mel_std + 1e-10)

        # align to EEG features
        words = downsampleLabels(words, eeg_sr, windowLength, frameshift)[:min_frames]
        words_trimmed = words[modelOrder * stepSize : words.shape[0] - modelOrder * stepSize]
        melSpec_trimmed = melSpec[modelOrder * stepSize : melSpec.shape[0] - modelOrder * stepSize, :]

        # validate exact frame alignment
        assert feat_stacked.shape[0] == melSpec_trimmed.shape[0] == words_trimmed.shape[0], \
            f"Shape mismatch for {participant}: feat = {feat_stacked.shape[0]}, spec = {melSpec_trimmed.shape[0]}, words = {words_trimmed.shape[0]}"

        # create feature names by appending the temporal shift
        feature_names = nameVector(channels, modelOrder, stepSize)

        # save all extracted feature files, target files and normalisation metadata
        np.save(os.path.join(path_output, f'{participant}_feat.npy'), feat_stacked)
        np.save(os.path.join(path_output, f'{participant}_procWords.npy'), words_trimmed)
        np.save(os.path.join(path_output, f'{participant}_spec.npy'), melSpec_trimmed)
        mel_norm = np.stack([mel_mean, mel_std]) # row 0 = mean, row 1 = std, shape (2, 80)
        np.save(os.path.join(path_output, f'{participant}_mel_norm.npy'), mel_norm)
        np.save(os.path.join(path_output, f'{participant}_feat_names.npy'), feature_names)