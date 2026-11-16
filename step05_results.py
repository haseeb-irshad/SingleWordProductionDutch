import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import scipy.io.wavfile as wavfile
from scipy.stats import pearsonr, wilcoxon
from scipy.ndimage import uniform_filter1d
from pystoi import stoi
from utilities import reconstructWave as rW
from utilities import MelFilterBank as mel
from step01_data_loader import BIDSDataLoader
from step02_extract_features import extractMelSpecs

def load_wav_float(path, expected_sr):
    """
    Load a WAV file and convert it to a float32 waveform in the range [-1.0, 1.0]

    Parameters
    ----------
    path : str
        File path to the .wav file
    expected_sr : int
        Expected sampling rate in Hz
    
    Returns
    -------
    audio : ndarray
        Float32 waveform in the range [-1.0, 1.0] formatted with shape (samples,)
    
    Raises
    ------
    ValueError
        If the file's sampling rate does not match `expected_sr`
    """
    sr, audio = wavfile.read(path)

    if sr != expected_sr:
        raise ValueError(f"Sampling rate mismatch in {path}: expected {expected_sr} Hz, but got {sr} Hz")

    # convert int16 range [-32768, 32767] to float32 range [-1.0, 1.0]
    if np.issubdtype(audio.dtype, np.integer):
        return audio.astype(np.float32) / 32768.0
    
    return audio.astype(np.float32)

def expected_vocoder_lag(vocoder, trimmed_frames, hop_length, n_fft, hifigan_padding_frames = 5):
    """
    Compute the expected number of samples by which a vocoded waveform lags the original recording

    Each saved spectrogram frame j describes the window that begins at original sample
    (j + trimmed_frames) * hop_length. Griffin-Lim uses the same uncentred framing, so its output
    lags the original recording by exactly trimmed_frames * hop_length samples. HiFi-GAN was trained
    on centred frames and places the audio for each frame around the frame centre rather than its
    start, which adds approximately n_fft // 2 samples to the lag. SpeechBrain's HiFi-GAN also pads
    the spectrogram with hifigan_padding_frames copies of its first frame, so its output starts
    hifigan_padding_frames * hop_length samples early and the lag is shorter by that amount

    Parameters
    ----------
    vocoder : str
        Vocoder name, either 'griffinlim' or 'hifigan'
    trimmed_frames : int
        Number of frames trimmed from the start of the spectrogram due to temporal context stacking
    hop_length : int
        STFT frameshift in samples
    n_fft : int
        STFT window length in samples
    hifigan_padding_frames : int, default = 5
        Number of frames that SpeechBrain's HiFi-GAN generator adds to each end of the spectrogram
        (inference_padding in speechbrain.lobes.models.HifiGAN)
    
    Returns
    -------
    lag : int
        Expected lag in samples (original sample = vocoded sample + lag)
    
    Raises
    ------
    ValueError
        If `vocoder` is not 'griffinlim' or 'hifigan'
    """
    lag = trimmed_frames * hop_length
    if vocoder == 'hifigan':
        lag += n_fft // 2 - hifigan_padding_frames * hop_length
    elif vocoder != 'griffinlim':
        raise ValueError(f"Unknown vocoder '{vocoder}': expected 'griffinlim' or 'hifigan'")
    return lag

def estimate_vocoder_lag(reference, vocoded, expected_lag, search_radius = 1024, smooth_len = 160, decim = 16):
    """
    Estimate the actual lag between a reference waveform and a vocoded waveform by correlating their
    amplitude envelopes around an expected lag

    Envelopes are used instead of raw waveforms because neither vocoder reproduces the original phase,
    so waveform correlation is weak even at the correct lag. Envelopes are smoothed over smooth_len
    samples and downsampled by decim, giving a lag resolution of decim samples

    Parameters
    ----------
    reference : ndarray
        Original recording formatted with shape (samples,)
    vocoded : ndarray
        Vocoded waveform formatted with shape (samples,)
    expected_lag : int
        Expected lag in samples based on vocoder type and STFT parameters
    search_radius : int, default = 1024
        Number of samples to search around the expected lag for peak correlation
    smooth_len : int, default = 160
        Moving-average length in samples for envelope smoothing (160 samples = 10 ms at 16 kHz)
    decim : int, default = 16
        Envelope downsampling factor and lag resolution in samples (16 samples = 1 ms at 16 kHz)
    
    Returns
    -------
    best_lag : int
        Estimated lag in samples that maximises the correlation between the reference and vocoded envelopes
    """
    # compute amplitude envelopes using absolute value and uniform smoothing
    ref_env = uniform_filter1d(np.abs(reference), size = smooth_len)[::decim]
    voc_env = uniform_filter1d(np.abs(vocoded), size = smooth_len)[::decim]

    # candidate lags on the downsampled grid, centred on the expected lag
    candidates = np.arange(expected_lag - search_radius, expected_lag + search_radius + 1, decim)
    candidates = candidates[candidates >= 0]

    # Pearson correlation between the envelopes at each candidate lag
    scores = np.full(len(candidates), -np.inf)
    for i, lag in enumerate(candidates):
        start = lag // decim
        n = min(len(ref_env) - start, len(voc_env))
        if n > 1:
            scores[i] = np.corrcoef(ref_env[start : start + n], voc_env[:n])[0, 1]

    best_lag = int(candidates[np.argmax(scores)])

    # a best lag on the edge of the search range means the true lag may lie outside it
    if best_lag in (candidates[0], candidates[-1]):
        print(f"Warning: estimated lag {best_lag} is at the edge of the search range around {expected_lag}")

    return best_lag

def align_to_reference(reference, processed, lag):
    """
    Align a processed waveform to a reference waveform by removing the leading lag samples
    from the recording and truncating both signals to a common length

    Parameters
    ----------
    reference : ndarray
        Original recording formatted with shape (samples,)
    processed : ndarray
        Vocoded waveform formatted with shape (samples,)
    lag : int
        Lag in samples (original sample = processed sample + lag)
    
    Returns
    -------
    reference_aligned : ndarray
        Reference waveform aligned with processed, formatted with shape (n,)
    processed_aligned : ndarray
        Processed waveform truncated to the common length, formatted with shape (n,)
    
    Raises
    ------
    ValueError
        If `lag` is negative
    """
    if lag < 0:
        raise ValueError(f"Lag must be non-negative, but got {lag}")

    reference = reference[lag:]
    n = min(len(reference), len(processed))

    return reference[:n], processed[:n]

def calculate_stoi_score(reference, processed, sr):
    """
    Calculate the Short-Time Objective Intelligibility (STOI) score (range 0.0 to 1.0)
    between a reference waveform and a vocoded waveform

    Both inputs must already be time-aligned (see align_to_reference) because STOI compares
    the two signals frame by frame and any offset lowers the score

    Parameters
    ----------
    reference : ndarray
        Original recording formatted with shape (samples,)
    processed : ndarray
        Vocoded waveform formatted with shape (samples,)
    sr : int
        Sampling rate of both signals in Hz
    
    Returns
    -------
    stoi_score : float
        STOI score (higher is better, 1.0 = perfect intelligibility)
    """
    # truncate to a common length
    n = min(len(reference), len(processed))

    # compute STOI score (extended = False computes the standard STOI (Taal et al., 2011))
    return float(stoi(reference[:n], processed[:n], sr, extended = False))

def speech_frame_mask(audio, n_fft, hop_length, dyn_range_db = 40.0):
    """
    Mark the frames of a waveform that contain speech using an energy threshold

    Frames use the same uncentred framing as rW.stft and extractMelSpecs, so frame k covers samples
    [k * hop_length, k * hop_length + n_fft). A frame counts as speech if its energy lies within
    dyn_range_db of the loudest frame, which is the criterion STOI uses to discard silent frames
    (Taal et al., 2011)

    Parameters
    ----------
    audio : ndarray
        Waveform formatted with shape (samples,)
    n_fft : int
        Frame length in samples
    hop_length : int
        Frameshift in samples
    dyn_range_db : float, default = 40.0
        Frames quieter than the loudest frame by more than this many dB are treated as silence
    
    Returns
    -------
    mask : ndarray
        Boolean speech mask formatted with shape (frames,)
    """
    num_frames = (len(audio) - n_fft) // hop_length + 1
    if num_frames <= 0:
        return np.zeros(0, dtype = bool)

    # windowed frame energies in dB, using the same Hann window as the STFT
    window = np.hanning(n_fft + 1)[:-1]
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop_length]
    energy_db = 10.0 * np.log10((frames ** 2) @ (window ** 2) + 1e-12)

    mask = energy_db > np.max(energy_db) - dyn_range_db

    return mask

def calculate_mcd_score(ref_log_mel, test_log_mel, num_coeffs = 15, frame_mask = None):
    """
    Calculate the Mel-Cepstral Distance (MCD) in dB between two natural log-Mel spectrograms

    Mel-cepstra are obtained with an orthonormal DCT-II across the Mel bins. The energy coefficient
    c0 is excluded, which makes the score insensitive to overall loudness differences between the
    two signals. Values are comparable within this project but not directly with MCD values computed
    from SPTK mel-cepstra, which are often reported in the speech synthesis literature

    Parameters
    ----------
    ref_log_mel : ndarray
        Reference natural log-Mel spectrogram formatted with shape (windows, mel_bins)
    test_log_mel : ndarray
        Test natural log-Mel spectrogram formatted with shape (windows, mel_bins)
    num_coeffs : int, default = 15
        Number of cepstral coefficients compared (c1 to c{num_coeffs})
    frame_mask : ndarray, optional
        Boolean mask formatted with shape (windows,) that selects the frames to evaluate, such as the
        speech frames from speech_frame_mask. All frames are used if None
    
    Returns
    -------
    mcd_db : float
        Mean MCD across the selected frames in dB (lower is better, 0.0 = identical spectral envelopes)
        or NaN if no frames are selected
    
    Raises
    ------
    ValueError
        If the spectrograms have fewer than num_coeffs + 1 Mel bins
    """
    # check that there are enough Mel bins to extract the requested coefficients
    num_mel_bins = ref_log_mel.shape[1]
    if num_mel_bins < num_coeffs + 1:
        raise ValueError(f"Spectrogram has only {num_mel_bins} Mel bins, which is insufficient to extract {num_coeffs} coefficients")

    # truncate both spectrograms to the shorter length
    n = min(ref_log_mel.shape[0], test_log_mel.shape[0])
    if frame_mask is not None:
        n = min(n, len(frame_mask))
    ref_log_mel = ref_log_mel[:n, :]
    test_log_mel = test_log_mel[:n, :]

    # keep only the selected frames so that silence does not dominate the average
    if frame_mask is not None:
        ref_log_mel = ref_log_mel[frame_mask[:n]]
        test_log_mel = test_log_mel[frame_mask[:n]]
        if ref_log_mel.shape[0] == 0:
            return np.nan

    # convert log-Mel spectra to mel-cepstra with an orthonormal DCT-II across the Mel bins
    ref_cep = scipy.fft.dct(ref_log_mel, type = 2, axis = 1, norm = 'ortho')
    test_cep = scipy.fft.dct(test_log_mel, type = 2, axis = 1, norm = 'ortho')

    # keep c1 to c{num_coeffs} because c0 only reflects overall energy
    diff = ref_cep[:, 1 : num_coeffs + 1] - test_cep[:, 1 : num_coeffs + 1]

    # convert the per-frame cepstral distance to dB using the factor (10 / ln 10) * sqrt(2)
    frame_mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.sum(diff ** 2, axis = 1))

    mcd_db = float(np.mean(frame_mcd))

    return mcd_db

def compute_lsd(reference, processed, fftsize, overlap, frame_mask = None):
    """
    Compute the Log-Spectral Distance (LSD) in dB between an original recording and a vocoded waveform

    For each STFT frame, LSD is the root-mean-square difference between the two log-magnitude spectra
    across frequency bins, expressed in dB, and the result is the mean over frames. Spectra are compared
    on a logarithmic scale because perceived loudness is approximately logarithmic in sound intensity.
    The processed spectrum is first rescaled to the total energy of the reference, so that the score
    reflects differences in spectral shape rather than overall loudness

    Parameters
    ----------
    reference : ndarray
        Original recording formatted with shape (samples,), time-aligned with processed
    processed : ndarray
        Vocoded waveform formatted with shape (samples,)
    fftsize : int
        STFT window length / FFT size
    overlap : int
        STFT overlap factor
    frame_mask : ndarray, optional
        Boolean mask formatted with shape (frames,) that selects the STFT frames to evaluate, such as
        the speech frames from speech_frame_mask. All frames are used if None

    Returns
    -------
    lsd_db : float
        Mean Log-Spectral Distance in dB across the selected frames (lower is better, 0.0 = identical
        spectra) or NaN if no frames are selected
    """
    # compute STFT magnitude spectra
    S1 = np.abs(rW.stft(reference, fftsize = fftsize, overlap = overlap))
    S2 = np.abs(rW.stft(processed, fftsize = fftsize, overlap = overlap))

    # truncate both spectra and the frame mask to the shortest length
    n = min(S1.shape[0], S2.shape[0])
    if frame_mask is not None:
        n = min(n, len(frame_mask))
    S1 = S1[:n, :]
    S2 = S2[:n, :]

    # keep only the selected frames so that silence does not dominate the average
    if frame_mask is not None:
        S1 = S1[frame_mask[:n]]
        S2 = S2[frame_mask[:n]]
        if S1.shape[0] == 0:
            return np.nan

    # rescale the processed spectrum to the total energy of the reference so that loudness differences are ignored
    S2 = S2 * np.sqrt(np.sum(S1 ** 2) / (np.sum(S2 ** 2) + 1e-10))

    # per-frame RMS difference of the log10 magnitudes, converted to dB by the factor 20
    log_diff = np.log10(S1 + 1e-5) - np.log10(S2 + 1e-5)
    lsd_per_frame = 20.0 * np.sqrt(np.mean(log_diff ** 2, axis = 1))

    lsd_db = float(np.mean(lsd_per_frame))

    return lsd_db

# Destrieux labels (without the ctx_lh_ or ctx_rh_ prefix) that define speech-relevant cortex
STG_LABELS = {'g_temp_sup-lateral', 'g_temp_sup-plan_tempo', 'g_temp_sup-plan_polar', 'g_temp_sup-g_t_transv'}
SMC_LABELS = {'g_precentral', 'g_postcentral', 's_central'}
VENTRAL_SMC_LABELS = {'g_and_s_subcentral', 's_precentral-inf-part'}
STS_LABEL = 's_temporal_sup'

def count_stg_vsmc_electrodes(channels_df, electrodes_df, vsmc_z_max = 40.0, include_sts = False):
    """
    Count the electrodes in the superior temporal gyrus (STG) and ventral sensorimotor cortex (vSMC)
    using the Destrieux atlas labels in the channels DataFrame

    STG contacts are those labelled with any subdivision of the superior temporal gyrus, including
    Heschl's gyrus (G_temp_sup-G_T_transv). The Destrieux atlas does not divide the sensorimotor strip
    into ventral and dorsal parts, so vSMC contacts are those in the subcentral gyrus or the inferior
    precentral sulcus, together with contacts in the precentral gyrus, postcentral gyrus or central
    sulcus whose ACPC z-coordinate is at most vsmc_z_max. The default of 40 mm approximates the boundary
    between the face and larynx representations and the more dorsal hand area. Contacts in both
    hemispheres are counted and sensorimotor contacts without coordinates are excluded because their
    height is unknown

    Parameters
    ----------
    channels_df : pd.DataFrame
        Per-participant channel metadata, as returned by BIDSDataLoader.load_channel_metadata(),
        with 'name' and 'description' columns
    electrodes_df : pd.DataFrame
        Per-participant 3D electrode coordinates in ACPC space, as returned by
        BIDSDataLoader.load_electrode_coordinates(), with 'name' and 'z' columns
    vsmc_z_max : float, default = 40.0
        Highest ACPC z-coordinate in mm at which a sensorimotor contact counts as ventral
    include_sts : bool, default = False
        Whether contacts in the superior temporal sulcus (S_temporal_sup) are added to the STG count

    Returns
    -------
    stg_count : int
        Number of STG contacts, including STS contacts if include_sts is True
    vsmc_count : int
        Number of vSMC contacts
    """
    # strip the hemisphere prefix so that labels from both hemispheres match the same Destrieux name
    labels = channels_df['description'].astype(str).str.lower().str.replace(r'^ctx_[lr]h_', '', regex = True)

    stg_labels = (STG_LABELS | {STS_LABEL}) if include_sts else STG_LABELS
    stg_count = int(labels.isin(stg_labels).sum())

    # contacts in the subcentral gyrus and inferior precentral sulcus are ventral by definition
    ventral_count = int(labels.isin(VENTRAL_SMC_LABELS).sum())

    # contacts on the rest of the sensorimotor strip only count as ventral below the z threshold
    smc_names = channels_df.loc[labels.isin(SMC_LABELS), 'name']
    smc_z = electrodes_df.loc[electrodes_df['name'].isin(smc_names), 'z']
    lower_smc_count = int((smc_z <= vsmc_z_max).sum())

    vsmc_count = ventral_count + lower_smc_count

    return stg_count, vsmc_count

def compare_to_random_control(allRes, randomControl):
    """
    Compare each participant's mean reconstruction correlation with their circular-shift null distribution

    The observed statistic is the mean correlation across folds and spectral bins. Each random round is
    summarised in the same way, as its mean correlation across spectral bins, so that both values describe
    the whole spectrogram. The p-value follows Phipson and Smyth (2010) by adding one to the numerator and
    denominator, which reflects that the observed data are themselves one possible arrangement and ensures
    the p-value is never exactly zero

    Parameters
    ----------
    allRes : ndarray
        Reconstruction correlations formatted with shape (participants, folds, mel_bins)
    randomControl : ndarray
        Circular-shift null correlations formatted with shape (participants, numRands, mel_bins)
    
    Returns
    -------
    observed : ndarray
        Mean reconstruction correlation per participant formatted with shape (participants,)
    null_max : ndarray
        Highest null correlation per participant formatted with shape (participants,)
    null_95 : ndarray
        95th percentile of the null distribution per participant formatted with shape (participants,)
    p_values : ndarray
        One-sided empirical p-value per participant formatted with shape (participants,)
    """
    num_participants = allRes.shape[0]
    observed = np.full(num_participants, np.nan)
    null_max = np.full(num_participants, np.nan)
    null_95 = np.full(num_participants, np.nan)
    p_values = np.full(num_participants, np.nan)

    # only participants processed in step03 are tested, so skipped participants stay NaN
    processed = ~np.all(np.isnan(allRes), axis = (1, 2))
    if not np.any(processed):
        return observed, null_max, null_95, p_values
    
    observed[processed] = np.mean(allRes[processed], axis = (1, 2))

    # summarise each random round by its mean correlation across spectral bins
    null_means = np.mean(randomControl[processed], axis = 2) # shape: (processed participants, numRands)

    null_max[processed] = np.max(null_means, axis = 1)
    null_95[processed] = np.percentile(null_means, 95, axis = 1)

    # count the random rounds that reach the observed value, adding one to both counts so that p is never zero
    num_exceed = np.sum(null_means >= observed[processed][:, None], axis = 1)
    p_values[processed] = (num_exceed + 1) / (null_means.shape[1] + 1)

    return observed, null_max, null_95, p_values

def mel_bin_positions(freqs_hz, num_bins, sr):
    """
    Convert frequencies in Hz to fractional Mel bin positions on the Slaney Mel scale used in step02

    The Mel range from 0 Hz to the Nyquist frequency is divided into num_bins + 1 equal steps and
    bin i is centred on step i + 1. This matches the Slaney filterbank that SpeechBrain applies in
    step02 with f_min = 0 Hz and f_max = sr / 2

    Parameters
    ----------
    freqs_hz : array_like
        Frequencies in Hz to convert
    num_bins : int
        Number of Mel bins in the spectrogram (e.g. 80)
    sr : int
        Audio sampling rate in Hz (e.g. 16000)
    
    Returns
    -------
    positions : ndarray
        Fractional Mel bin positions formatted with shape (len(freqs_hz),)
    """
    # the STFT size passed to MelFilterBank does not affect its Mel scale conversion
    mfb = mel.MelFilterBank(513, num_bins, sr)
    mel_step = mfb.freqToMel(sr / 2.0) / (num_bins + 1)

    positions = np.array([mfb.freqToMel(f) / mel_step - 1 for f in freqs_hz])

    return positions

def paired_vocoder_test(gl_scores, hifi_scores):
    """
    Compare Griffin-Lim and HiFi-GAN scores across participants with a two-sided Wilcoxon signed-rank test

    Each participant contributes one score per vocoder, so the comparison is paired. The Wilcoxon
    signed-rank test is used instead of a paired t-test because the normality of the paired differences
    cannot be checked reliably with around ten participants. Only participants with a valid score for both
    vocoders are included

    Parameters
    ----------
    gl_scores : array_like
        Griffin-Lim score per participant formatted with shape (participants,) with NaN for missing values
    hifi_scores : array_like
        HiFi-GAN score per participant formatted with shape (participants,) with NaN for missing values
    
    Returns
    -------
    n_pairs : int
        Number of participants with valid scores for both vocoders
    median_diff : float
        Median of the paired differences (HiFi-GAN minus Griffin-Lim) or NaN if there are no pairs
    p_value : float
        Two-sided p-value or NaN if fewer than two pairs remain or all differences are zero
    """
    gl_scores = np.asarray(gl_scores, dtype = float)
    hifi_scores = np.asarray(hifi_scores, dtype = float)

    # keep only participants with a valid score for both vocoders
    paired = ~np.isnan(gl_scores) & ~np.isnan(hifi_scores)
    diffs = hifi_scores[paired] - gl_scores[paired]
    n_pairs = int(np.sum(paired))

    if n_pairs == 0:
        return n_pairs, np.nan, np.nan

    median_diff = float(np.median(diffs))

    # the test is undefined for a single pair or when every difference is zero
    if n_pairs < 2 or np.all(diffs == 0):
        return n_pairs, median_diff, np.nan

    _, p_value = wilcoxon(diffs)

    return n_pairs, median_diff, float(p_value)

def holm_correction(p_values):
    """
    Adjust p-values for multiple comparisons with the Holm-Bonferroni method

    The p-values are sorted in ascending order and the k-th smallest (counting from zero) is multiplied
    by (m - k), where m is the number of valid p-values. The adjusted values are then made
    non-decreasing in the same order and capped at 1.0. NaN p-values are ignored and returned as NaN

    Parameters
    ----------
    p_values : array_like
        Uncorrected p-values formatted with shape (tests,)
    
    Returns
    -------
    adjusted : ndarray
        Holm-corrected p-values formatted with shape (tests,)
    """
    p_values = np.asarray(p_values, dtype = float)
    adjusted = np.full(p_values.shape, np.nan)

    valid = ~np.isnan(p_values)
    p_valid = p_values[valid]
    m = len(p_valid)
    order = np.argsort(p_valid)

    # scale each sorted p-value, keep the sequence non-decreasing and cap it at 1.0
    adj_sorted = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p_valid[order]), 1.0)

    # return the adjusted values to their original order
    adj_valid = np.empty(m)
    adj_valid[order] = adj_sorted
    adjusted[valid] = adj_valid

    return adjusted

def plot_benchmark_metric(ax, gl_data, hifi_data, ylabel_str, title_str, palette, p_value = None):
    """
    Plot paired Griffin-Lim and HiFi-GAN scores as bars with each participant's two points joined by a line

    Only participants with a valid score for both vocoders are shown, so that both bars average the same
    participants and every point has a partner. Bars show the mean and error bars show the standard error
    of the mean

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    gl_data : array_like
        Griffin-Lim score per participant formatted with shape (participants,) with NaN for missing values
    hifi_data : array_like
        HiFi-GAN score per participant formatted with shape (participants,) with NaN for missing values
    ylabel_str : str
        Y-axis label
    title_str : str
        Panel letter shown as the title
    palette : list of str
        Bar colours for Griffin-Lim and HiFi-GAN
    p_value : float, optional
        Holm-corrected Wilcoxon p-value shown above the bars
    """
    gl_data = np.asarray(gl_data, dtype = float)
    hifi_data = np.asarray(hifi_data, dtype = float)

    # keep only participants with a valid score for both vocoders
    paired = ~np.isnan(gl_data) & ~np.isnan(hifi_data)
    if not np.any(paired):
        ax.text(0.5, 0.5, 'No paired evaluation results found', ha = 'center', va = 'center', transform = ax.transAxes)
        return

    gl_vals = gl_data[paired]
    hifi_vals = hifi_data[paired]
    n_pairs = len(gl_vals)

    # the standard error of the mean uses the sample standard deviation (ddof = 1)
    means = [np.mean(gl_vals), np.mean(hifi_vals)]
    sems = [np.std(vals, ddof = 1) / np.sqrt(n_pairs) if n_pairs > 1 else 0.0 for vals in (gl_vals, hifi_vals)]
    ax.bar([0, 1], means, yerr = sems, color = palette, alpha = 0.8, capsize = 6, edgecolor = 'k', linewidth = 0.8)

    # join each participant's two scores with a line
    for gl_val, hifi_val in zip(gl_vals, hifi_vals):
        ax.plot([0, 1], [gl_val, hifi_val], color = 'gray', alpha = 0.4, linestyle = '--')
    ax.scatter(np.zeros(n_pairs), gl_vals, color = 'black', alpha = 0.7, zorder = 3, s = 30)
    ax.scatter(np.ones(n_pairs), hifi_vals, color = 'black', alpha = 0.7, zorder = 3, s = 30)

    # show the Holm-corrected Wilcoxon p-value above the bars
    if p_value is not None and not np.isnan(p_value):
        ax.text(0.5, 0.98, f'p = {p_value:.3f} (n = {n_pairs})', transform = ax.transAxes, ha = 'center', va = 'top', fontsize = 12)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Griffin-Lim', 'HiFi-GAN'], fontsize = 12)
    ax.set_ylabel(ylabel_str, fontsize = 13)
    ax.set_title(title_str, fontsize = 20)

def find_word_intervals(word_frames):
    """
    Find the runs of consecutive frames labelled with the same word

    In this dataset each run covers the period in which the word was shown on screen, so the participant
    speaks somewhere within it rather than at its start

    Parameters
    ----------
    word_frames : ndarray
        Word label per spectrogram frame formatted with shape (frames,) with '' for frames without a word
    
    Returns
    -------
    intervals : list of tuple
        (word, start_frame, end_frame) for each run, where end_frame is exclusive
    """
    intervals = []
    i = 0
    num_frames = len(word_frames)

    while i < num_frames:
        word = word_frames[i]
        if word == '':
            i += 1
            continue

        # advance to the end of this run of identical labels
        start = i
        while i < num_frames and word_frames[i] == word:
            i += 1
        intervals.append((word, start, i))

    return intervals

def log_magnitude_db(audio, fftsize, overlap, dynamic_range_db = 80.0):
    """
    Compute a log-magnitude STFT spectrogram in dB relative to its own maximum

    Each waveform was peak-normalised separately in step02 and step04, so levels are expressed relative
    to each spectrogram's own maximum. Values more than dynamic_range_db below the maximum are clipped so
    that the silence between words does not stretch the colour scale

    Parameters
    ----------
    audio : ndarray
        Waveform formatted with shape (samples,)
    fftsize : int
        STFT window length / FFT size
    overlap : int
        STFT overlap factor
    dynamic_range_db : float, default = 80.0
        Range in dB below the maximum that is kept
    
    Returns
    -------
    spec_db : ndarray
        Log-magnitude spectrogram in dB relative to its maximum formatted with shape (frames, fftsize // 2 + 1)
    """
    spec = np.abs(rW.stft(audio, fftsize = fftsize, overlap = overlap))
    spec_db = 20.0 * np.log10(spec + 1e-10)
    spec_db = spec_db - np.max(spec_db)

    spec_db = np.maximum(spec_db, -dynamic_range_db)

    return spec_db

if __name__ == "__main__":
    # build paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_bids = os.path.join(script_dir, 'SingleWordProductionDutch-iBIDS')
    result_path = os.path.join(script_dir, 'results')
    feat_path = os.path.join(script_dir, 'features')
    audio_path = os.path.join(result_path, 'audio')

    # directory setup for saving figures
    figures_path = os.path.join(result_path, 'figures')
    os.makedirs(figures_path, exist_ok = True)

    participants = [f'sub-{i+1:02d}' for i in range(10)]
    colors = ['C' + str(i) for i in range(10)]

    allRes = np.load(os.path.join(result_path, 'linearResults.npy'))
    randomControl = np.load(os.path.join(result_path, 'randomResults.npy'))
    explainedVariance = np.load(os.path.join(result_path, 'explainedVariance.npy'))

    # compare each participant's mean correlation with their random-shift distribution
    meanCorrs, null_max, null_95, p_values = compare_to_random_control(allRes, randomControl)

    # participants processed in step03, since skipped participants are stored as NaN
    processed = ~np.all(np.isnan(allRes), axis = (1, 2))

    # sample standard deviation across folds and spectral bins for the whiskers in panel a
    stdCorrs = np.full(len(participants), np.nan)
    stdCorrs[processed] = np.std(allRes[processed], axis = (1, 2), ddof = 1)

    # explained variance of the retained PCA components averaged across folds, as a percentage
    meanExpVar = np.full(len(participants), np.nan)
    meanExpVar[processed] = np.mean(explainedVariance[processed], axis = 1) * 100

    numRands = randomControl.shape[1]
    print(f"\nReconstruction results against {numRands} random circular shifts")
    print(f"{'Participant':<12} | {'Mean r':<7} | {'Null 95%':<8} | {'Null max':<8} | {'p':<7} | {'PCA var (%)':<11}")
    for p, sub in enumerate(participants):
        if not processed[p]:
            print(f"{sub:<12} | not processed in step03")
            continue
        print(f"{sub:<12} | {meanCorrs[p]:<7.3f} | {null_95[p]:<8.3f} | {null_max[p]:<8.3f} | {p_values[p]:<7.4f} | {meanExpVar[p]:<11.1f}")

    print(f"Participants above all random shifts: {np.sum(meanCorrs[processed] > null_max[processed])} of {np.sum(processed)}")
    print(f"PCA explained variance across participants: {np.min(meanExpVar[processed]):.1f}% to {np.max(meanExpVar[processed]):.1f}%")

    target_sr = 16000   # expected audio sampling rate in Hz
    fftsize = 1024      # STFT window length / FFT size
    overlap = 4         # STFT overlap factor
    hop_length = 256    # STFT frameshift in samples

    # frames removed from the start of the spectrogram by the temporal context stacking
    modelOrder = 4
    stepSize = 3
    trimmed_frames = modelOrder * stepSize

    x = range(len(meanCorrs))
    fig, ax = plt.subplots(1, 2, figsize = (14, 7))

    # panel a: mean correlation per participant with fold-level scatter points
    ax[0].bar(x, meanCorrs, yerr = stdCorrs, alpha = 0.5, color = colors)
    for p in range(allRes.shape[0]):
        if not processed[p]:
            continue

        # mean correlation of each fold as scatter points
        ax[0].scatter(
            np.zeros(allRes[p, :, :].shape[0]) + p,
            np.nanmean(allRes[p, :, :], axis = 1),
            color = colors[p]
        )

    # mark the highest random-shift correlation for each participant as their chance level
    bar_positions = np.arange(len(null_max))
    ax[0].hlines(null_max, bar_positions - 0.4, bar_positions + 0.4, colors = 'k', linestyles = 'dashed', linewidth = 1.5, label = 'Chance (max of random shifts)')
    ax[0].legend(fontsize = 12, frameon = False)

    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f'sub-{i+1:02d}' for i in x], rotation = 45, ha = 'right', fontsize = 16)
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel('Correlation', fontsize = 16)
    ax[0].set_title('a', fontsize = 20)

    plt.setp(ax[0].spines.values(), linewidth = 2)
    ax[0].xaxis.set_tick_params(width = 2)
    ax[0].yaxis.set_tick_params(width = 2)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    # panel b: mean correlation across folds for each Mel bin
    specMean = np.nanmean(allRes[processed], axis = 1)
    specStd = np.std(allRes[processed], axis = 1, ddof = 1)
    specBins = np.arange(1, allRes.shape[2] + 1)

    # standard error of the mean across folds, shown as shaded areas
    specErr = specStd / np.sqrt(allRes.shape[1])

    for row, p in enumerate(np.where(processed)[0]):
        ax[1].plot(specBins, specMean[row, :], color = colors[p])
        ax[1].fill_between(specBins, specMean[row, :] - specErr[row, :], specMean[row, :] + specErr[row, :], alpha = 0.5, color = colors[p])

    # the highest random-shift correlation in each bin across all participants gives a conservative chance level
    bin_null_max = np.nanmax(randomControl, axis = (0, 1))
    ax[1].plot(specBins, bin_null_max, color = 'k', linestyle = '--', linewidth = 1.5, label = 'Chance (max of random shifts, all participants)')
    ax[1].legend(fontsize = 12, frameon = False)

    ax[1].set_ylim(0, 1)
    ax[1].set_xlim(1, len(specBins))
    ax[1].set_xticks([1] + list(range(10, len(specBins) + 1, 10)))
    ax[1].set_xlabel('Mel Bin', fontsize = 16)
    ax[1].set_ylabel('Correlation', fontsize = 16)
    ax[1].set_title('b', fontsize = 20)

    plt.setp(ax[1].spines.values(), linewidth = 2)
    ax[1].xaxis.set_tick_params(width = 2)
    ax[1].yaxis.set_tick_params(width = 2)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'results.png'), dpi = 600)
    plt.show()

    # spatial channel contributions and anatomical coverage
    fig, ax_chan = plt.subplots(1, 2, figsize = (14, 6))

    # left panel: decoder weight distribution for the participants with the highest and lowest mean correlation
    processed_idx = np.where(processed)[0]
    performers = []
    if len(processed_idx) >= 2:
        best_idx = processed_idx[np.argmax(meanCorrs[processed_idx])]
        worst_idx = processed_idx[np.argmin(meanCorrs[processed_idx])]
        performers = [(best_idx, 'highest'), (worst_idx, 'lowest')]

    # channel importance arrays are saved by step03
    importance_paths = {idx: os.path.join(result_path, f'{participants[idx]}_channel_importance.npy') for idx, _ in performers}

    if len(performers) == 2 and all(os.path.exists(path) for path in importance_paths.values()):
        for idx, rank_label in performers:
            chans = np.load(importance_paths[idx])

            # express each channel's weight magnitude as a percentage of the participant's total and sort in descending order
            chans_sorted = np.sort(chans / np.sum(chans) * 100)[::-1]

            # map channel ranks to percentiles so that participants with different channel counts are comparable
            rank_percent = np.linspace(0, 100, len(chans_sorted))

            ax_chan[0].plot(rank_percent, chans_sorted, color = colors[idx], lw = 2.5, label = f'{participants[idx]} ({rank_label} mean r = {meanCorrs[idx]:.2f})')

        # shade the top 10% of channels
        ax_chan[0].axvspan(0, 10, color = 'gray', alpha = 0.15, label = 'Top 10% Channels')

        ax_chan[0].set_xlabel('Electrode Channel Rank (% of total channels)', fontsize = 14)
        ax_chan[0].set_ylabel('Share of Decoder Weight (%)', fontsize = 14)
        ax_chan[0].set_title('Decoder Weight Distribution Across Channels', fontsize = 16)
        ax_chan[0].legend(fontsize = 12, frameon = False)
        ax_chan[0].set_xlim(0, 100)
        ax_chan[0].set_ylim(bottom = 0)
    else:
        ax_chan[0].text(0.5, 0.5, 'Channel importance files not found or fewer than two participants processed.\nRun step03_train_decoder.py first',
                        ha = 'center', va = 'center', transform = ax_chan[0].transAxes, fontsize = 12)

    plt.setp(ax_chan[0].spines.values(), linewidth = 2)
    ax_chan[0].xaxis.set_tick_params(width = 2)
    ax_chan[0].yaxis.set_tick_params(width = 2)
    ax_chan[0].spines['right'].set_visible(False)
    ax_chan[0].spines['top'].set_visible(False)

    # right panel: decoding performance against the number of electrodes in speech-relevant cortex (STG and vSMC)
    loader = BIDSDataLoader(bids_root = path_bids)
    region_counts = np.array([count_stg_vsmc_electrodes(loader.load_channel_metadata(sub), loader.load_electrode_coordinates(sub)) for sub in participants]) # shape: (participants, 2)
    speech_cortex_electrodes = region_counts.sum(axis = 1)

    print(f"\n{'Participant':<12} | {'STG':<4} | {'vSMC':<4} | {'Total':<5}")
    for p, sub in enumerate(participants):
        print(f"{sub:<12} | {region_counts[p, 0]:<4} | {region_counts[p, 1]:<4} | {speech_cortex_electrodes[p]:<5}")
    
    # only processed participants enter the scatter, the correlation and the trendline
    for idx in processed_idx:
        ax_chan[1].scatter(speech_cortex_electrodes[idx], meanCorrs[idx], color = colors[idx], s = 120, zorder = 4, edgecolor = 'k', linewidth = 0.8, label = participants[idx])

    # a correlation and trendline need at least three participants to be meaningful
    if len(processed_idx) >= 3:
        r_val, p_val = pearsonr(speech_cortex_electrodes[processed], meanCorrs[processed])
        m, c = np.polyfit(speech_cortex_electrodes[processed], meanCorrs[processed], 1)
        x_fit = np.linspace(max(0, speech_cortex_electrodes[processed].min() - 1), speech_cortex_electrodes[processed].max() + 1, 100)
        ax_chan[1].plot(x_fit, m * x_fit + c, color = 'gray', linestyle = '--', lw = 2, label = f'Linear Fit (r = {r_val:.2f}, p = {p_val:.3f})')

    # participant colours match the results figure, so a legend replaces overlapping text labels
    ax_chan[1].legend(fontsize = 10, loc = 'lower right', ncol = 2, frameon = False)

    ax_chan[1].set_xlabel('Number of STG / vSMC Electrodes', fontsize = 14)
    ax_chan[1].set_ylabel('Mean Decoding Correlation (r)', fontsize = 14)
    ax_chan[1].set_title('Performance vs Speech Cortex Coverage', fontsize = 16)
    ax_chan[1].set_ylim(0, 1)

    plt.setp(ax_chan[1].spines.values(), linewidth = 2)
    ax_chan[1].xaxis.set_tick_params(width = 2)
    ax_chan[1].yaxis.set_tick_params(width = 2)
    ax_chan[1].spines['right'].set_visible(False)
    ax_chan[1].spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'channel_coverage_analysis.png'), dpi = 600)
    plt.show()

    # benchmark Griffin-Lim and HiFi-GAN under the oracle and decoded conditions
    # the oracle condition isolates the vocoder loss, whereas the decoded condition also includes the decoding loss
    vocoders = ['griffinlim', 'hifigan']
    conditions = ['oracle', 'decoded']
    condition_labels = {'oracle': 'Oracle (ground-truth spectrogram)', 'decoded': 'Decoded (predicted spectrogram)'}

    stoi_scores = {condition: {voc: [] for voc in vocoders} for condition in conditions}
    lsd_scores = {condition: {voc: [] for voc in vocoders} for condition in conditions}
    mcd_audio_scores = {condition: {voc: [] for voc in vocoders} for condition in conditions}
    mcd_spec_list = []

    # estimated lag per participant and vocoder
    vocoder_lags = {}

    eval_participants = []

    for sub in participants:
        gt_spec_f = os.path.join(feat_path, f'{sub}_spec.npy')
        pred_spec_f = os.path.join(result_path, f'{sub}_predicted_spec.npy')
        orig_wav_f = os.path.join(feat_path, f'{sub}_orig_audio.wav')

        if not os.path.exists(gt_spec_f) or not os.path.exists(pred_spec_f) or not os.path.exists(orig_wav_f):
            print(f"Skipping {sub}: missing spectrogram or original audio file")
            continue

        eval_participants.append(sub)
        gt_spec = np.load(gt_spec_f)
        pred_spec = np.load(pred_spec_f)

        # the original recording is the reference for both vocoders
        orig_audio = load_wav_float(orig_wav_f, target_sr)

        # spectrogram-level MCD measures decoding accuracy alone because no vocoder is involved
        norm_path = os.path.join(feat_path, f'{sub}_mel_norm.npy')
        if os.path.exists(norm_path):
            mel_norm = np.load(norm_path)
            mel_mean, mel_std = mel_norm[0], mel_norm[1]
            gt_log_mel = (gt_spec * mel_std) + mel_mean
            pred_log_mel = (pred_spec * mel_std) + mel_mean

            # spectrogram frame j covers original samples from (trimmed_frames + j) * hop_length onwards
            spec_speech_mask = speech_frame_mask(orig_audio[trimmed_frames * hop_length :], fftsize, hop_length)

            print(f"{sub}: {100 * np.mean(spec_speech_mask):.1f}% of frames classified as speech")
            mcd_spec_list.append(calculate_mcd_score(gt_log_mel, pred_log_mel, frame_mask = spec_speech_mask))
        else:
            mcd_spec_list.append(np.nan)
        
        vocoder_lags[sub] = {}

        for voc in vocoders:
            gt_voc_f = os.path.join(audio_path, f'{sub}_ground_truth_{voc}.wav')
            pred_voc_f = os.path.join(audio_path, f'{sub}_predicted_{voc}.wav')

            if not os.path.exists(gt_voc_f) or not os.path.exists(pred_voc_f):
                for condition in conditions:
                    stoi_scores[condition][voc].append(np.nan)
                    lsd_scores[condition][voc].append(np.nan)
                    mcd_audio_scores[condition][voc].append(np.nan)
                continue

            gt_voc = load_wav_float(gt_voc_f, target_sr)
            pred_voc = load_wav_float(pred_voc_f, target_sr)

            # estimate the lag on the ground-truth resynthesis, then apply it to the prediction
            expected_lag = expected_vocoder_lag(voc, trimmed_frames, hop_length, fftsize)
            lag = estimate_vocoder_lag(orig_audio, gt_voc, expected_lag)
            vocoder_lags[sub][voc] = lag
            print(f"{sub} {voc}: expected lag = {expected_lag} samples, estimated lag = {lag} samples")

            # both files come from the same vocoder and frame timing, so they share the lag
            vocoded = {'oracle': gt_voc, 'decoded': pred_voc}

            # compare each vocoded waveform with the time-aligned original recording
            for condition in conditions:
                ref_aligned, voc_aligned = align_to_reference(orig_audio, vocoded[condition], lag)

                # speech frames come from the original recording so that every method is scored on the same frames
                ref_speech_mask = speech_frame_mask(ref_aligned, fftsize, hop_length)

                stoi_scores[condition][voc].append(calculate_stoi_score(ref_aligned, voc_aligned, target_sr))
                lsd_scores[condition][voc].append(compute_lsd(ref_aligned, voc_aligned, fftsize, overlap, frame_mask = ref_speech_mask))

                # audio-level MCD re-analyses both waveforms with the step02 Mel pipeline
                ref_log_mel = extractMelSpecs(ref_aligned, target_sr)
                voc_log_mel = extractMelSpecs(voc_aligned, target_sr)
                mcd_audio_scores[condition][voc].append(calculate_mcd_score(ref_log_mel, voc_log_mel, frame_mask = ref_speech_mask))

    # short vocoder names and the metrics shown in each table
    vocoder_abbrev = {'griffinlim': 'GL', 'hifigan': 'HiFi'}
    table_metrics = [('STOI', stoi_scores), ('LSD', lsd_scores), ('MCD', mcd_audio_scores)]

    # one table per condition
    for condition in conditions:
        print(f"\n{condition_labels[condition]} (LSD and MCD in dB, speech frames only)")
        columns = [f"{name} {vocoder_abbrev[voc]}" for name, _ in table_metrics for voc in vocoders]

        # spectrogram MCD only applies to the decoded condition because it compares ground-truth and predicted spectrograms
        if condition == 'decoded':
            columns.append('MCD spec')
        print(f"{'Participant':<12} | " + " | ".join(f"{col:<9}" for col in columns))

        for i, sub in enumerate(eval_participants):
            values = [scores[condition][voc][i] for _, scores in table_metrics for voc in vocoders]
            if condition == 'decoded':
                values.append(mcd_spec_list[i])
            print(f"{sub:<12} | " + " | ".join(f"{val:<9.3f}" for val in values))

    # paired Wilcoxon signed-rank tests of HiFi-GAN against Griffin-Lim for every metric and condition
    test_keys, test_results = [], []
    for name, scores in table_metrics:
        for condition in conditions:
            test_keys.append((name, condition))
            test_results.append(paired_vocoder_test(scores[condition]['griffinlim'], scores[condition]['hifigan']))

    # correct the p-values for testing three metrics in two conditions
    holm_p = holm_correction([result[2] for result in test_results])
    p_adjusted = dict(zip(test_keys, holm_p))

    print(f"\nHiFi-GAN vs Griffin-Lim (Wilcoxon signed-rank, Holm-corrected across {len(test_keys)} tests)")
    print("Positive differences favour HiFi-GAN for STOI and negative differences favour HiFi-GAN for LSD and MCD")
    print(f"{'Metric':<6} | {'Condition':<9} | {'n':<3} | {'Median diff':<11} | {'p':<7} | {'p (Holm)':<8}")
    for (name, condition), (n_pairs, median_diff, p_value), p_holm in zip(test_keys, test_results, holm_p):
        print(f"{name:<6} | {condition:<9} | {n_pairs:<3} | {median_diff:<11.3f} | {p_value:<7.4f} | {p_holm:<8.4f}")
    
    if len(eval_participants) > 0:
        figure_metrics = [('STOI', stoi_scores, 'STOI Score (0.0 - 1.0)'), ('LSD', lsd_scores, 'LSD (dB)'), ('MCD', mcd_audio_scores, 'MCD (dB)')]
        fig, axes = plt.subplots(len(figure_metrics), 2, figsize = (12, 14), sharey = 'row')
        palette = ['#7293CB', '#E1974C']
        panel_letters = 'abcdef'

        for row, (name, scores, ylabel_str) in enumerate(figure_metrics):
            for col, condition in enumerate(conditions):
                plot_benchmark_metric(axes[row, col], scores[condition]['griffinlim'], scores[condition]['hifigan'], ylabel_str, panel_letters[2 * row + col], palette, p_value = p_adjusted[(name, condition)])

                # label each panel with its condition below the vocoder tick labels
                axes[row, col].set_xlabel(condition_labels[condition], fontsize = 12)

            # the shared y-axis makes the right-hand label redundant
            axes[row, 1].set_ylabel('')
        
        # STOI is bounded between 0 and 1, so the extra room above 1 only holds the p-value labels
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].set_yticks(np.arange(0, 1.01, 0.2))

        # leave room above the data for the p-value labels in the rows without a fixed range
        for row in range(1, len(figure_metrics)):
            axes[row, 0].set_ylim(0, axes[row, 0].get_ylim()[1] * 1.15)

        for ax in axes.flat:
            plt.setp(ax.spines.values(), linewidth = 1.5)
            ax.xaxis.set_tick_params(width = 1.5)
            ax.yaxis.set_tick_params(width = 1.5)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, 'acoustic_benchmarking_hifigan_vs_gl.png'), dpi = 600)
        plt.show()

    # qualitative spectrogram reconstruction and word alignment
    participant = 'sub-06'

    # timeframe to plot in seconds of the original recording
    start_s = 5.8
    stop_s = 19.8
    frameshift = hop_length / target_sr

    # spectrogram frame j starts at original sample orig_offset + j * hop_length
    orig_offset = trimmed_frames * hop_length
    pSta = int(round((start_s * target_sr - orig_offset) / hop_length))
    pSto = int(round((stop_s * target_sr - orig_offset) / hop_length))

    # exact start sample and times of the plotted frames after rounding to whole frames
    win_start_sample = orig_offset + pSta * hop_length
    win_start_s = win_start_sample / target_sr
    win_stop_s = win_start_s + (pSto - pSta) * frameshift

    # load spectrograms and processed words
    rec_spec = np.load(os.path.join(result_path, f'{participant}_predicted_spec.npy'))
    spectrogram = np.load(os.path.join(feat_path, f'{participant}_spec.npy'))
    proc_words = np.load(os.path.join(feat_path, f'{participant}_procWords.npy'))

    # cut the original recording to the same stretch of time as the plotted frames
    rate = target_sr
    orig_full = load_wav_float(os.path.join(feat_path, f'{participant}_orig_audio.wav'), rate)
    orig = orig_full[win_start_sample : win_start_sample + (pSto - pSta) * hop_length]

    # word display intervals within the plotted window, converted to seconds of the original recording
    word_intervals = [(word, win_start_s + start * frameshift, win_start_s + end * frameshift) for word, start, end in find_word_intervals(proc_words[pSta : pSto])]

    # both panels use the colour range of the ground truth so that the reconstruction is not rescaled to look sharper
    gt_window = spectrogram[pSta : pSto, :]
    rec_window = rec_spec[pSta : pSto, :]
    vmin, vmax = np.percentile(gt_window, [1, 99])

    # label the Mel bin axis with frequencies in Hz
    freq_ticks_hz = [250, 500, 1000, 2000, 4000]
    freq_ticks_pos = mel_bin_positions(freq_ticks_hz, gt_window.shape[1], target_sr)

    # place the images on a time axis in seconds of the original recording
    extent = [win_start_s, win_stop_s, -0.5, gt_window.shape[1] - 0.5]

    fig, ax_spec = plt.subplots(2, sharex = True, figsize = (12, 6), constrained_layout = True)

    for ax, data, title_str in zip(ax_spec, [gt_window, rec_window], ['a', 'b']):
        im = ax.imshow(data.T, origin = 'lower', aspect = 'auto', cmap = 'viridis', vmin = vmin, vmax = vmax, extent = extent)
        ax.set_yticks(freq_ticks_pos)
        ax.set_yticklabels([str(f) for f in freq_ticks_hz])
        ax.set_ylabel('Frequency (Hz)', fontsize = 14)
        ax.set_title(title_str, fontsize = 20)

    # mark when each word was shown on screen with its label at the top of panel a
    for ax in ax_spec:
        for word, word_start, word_end in word_intervals:
            ax.axvline(word_start, color = 'white', linestyle = '--', lw = 1)
            ax.axvline(word_end, color = 'white', linestyle = '--', lw = 1)

    for word, word_start, word_end in word_intervals:
        ax_spec[0].text((word_start + word_end) / 2, 0.95, word, transform = ax_spec[0].get_xaxis_transform(), color = 'white', ha = 'center', va = 'top', fontsize = 11)

    ax_spec[1].set_xlabel('Time in recording (s)', fontsize = 14)

    # one colour bar serves both panels because they share the same colour range
    fig.colorbar(im, ax = ax_spec, label = 'Z-scored log-Mel magnitude')

    for ax in ax_spec:
        plt.setp(ax.spines.values(), linewidth = 2)
        ax.xaxis.set_tick_params(width = 2)
        ax.yaxis.set_tick_params(width = 2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.savefig(os.path.join(figures_path, 'spec_example.png'), dpi = 600)
    plt.show()

    # compare the original recording with the Griffin-Lim and HiFi-GAN waveforms over the same stretch of time
    waveforms = [('Original\nrecording', orig, '#1f77b4')]
    tracks = [('griffinlim', 'Griffin-Lim\n(decoded)', '#ff7f0e'), ('hifigan', 'HiFi-GAN\n(decoded)', '#2ca02c')]

    for voc, label_text, color in tracks:
        voc_wav_f = os.path.join(audio_path, f'{participant}_predicted_{voc}.wav')
        if not os.path.exists(voc_wav_f):
            continue
        voc_audio = load_wav_float(voc_wav_f, target_sr)

        # vocoded sample n matches original sample n + lag using the expected lag if this participant has no estimate
        expected_lag = expected_vocoder_lag(voc, trimmed_frames, hop_length, fftsize)
        lag = vocoder_lags.get(participant, {}).get(voc, expected_lag)

        voc_start = win_start_sample - lag
        if voc_start < 0:
            raise ValueError(f"{label_text} audio does not cover the start of the plotted window, so choose a later start_s")
        waveforms.append((label_text, voc_audio[voc_start : voc_start + len(orig)], color))
    
    num_plots = len(waveforms)
    fig, axarr = plt.subplots(num_plots, 1, sharex = True, figsize = (12, 2.5 * num_plots))
    if num_plots == 1:
        axarr = [axarr]

    panel_labels = ['a', 'b', 'c']

    # time of each sample in seconds of the original recording
    time_s = win_start_s + np.arange(len(orig)) / rate

    for idx, (label_text, wav_data, color) in enumerate(waveforms):
        ax = axarr[idx]

        # shade the intervals in which each word was shown on screen
        for word, word_start, word_end in word_intervals:
            ax.axvspan(word_start, word_end, color = 'gray', alpha = 0.15, lw = 0)
        
        ax.plot(time_s[:len(wav_data)], wav_data, color = color, alpha = 0.85, lw = 0.8)
        ax.set_ylabel(label_text, fontsize = 12)
        ax.set_title(panel_labels[idx], fontsize = 16)

        # scale each panel to its own peak because every file was peak-normalised separately
        max_amp = np.max(np.abs(wav_data)) if np.max(np.abs(wav_data)) > 0 else 1.0
        ax.set_ylim([-1.35 * max_amp, 1.35 * max_amp])
        ax.set_yticks([])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # only the bottom panel keeps its time axis
        if idx < num_plots - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(bottom = False)

    # label each shaded interval with its word at the top of the first panel
    for word, word_start, word_end in word_intervals:
        axarr[0].text((word_start + word_end) / 2, 0.95, word, transform = axarr[0].get_xaxis_transform(), ha = 'center', va = 'top', fontsize = 11)

    bottom_ax = axarr[-1]
    bottom_ax.set_xlim([time_s[0], time_s[-1]])
    bottom_ax.set_xlabel('Time in recording (s)', fontsize = 12)
    bottom_ax.tick_params(width = 2)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'wav_example_comparison.png'), dpi = 600)
    plt.show()

    # close-up spectrograms of one word from the original recording and both vocoders under both conditions
    complete_words = [interval for interval in word_intervals if interval[1] > win_start_s and interval[2] < win_stop_s]
    closeup_files = {(condition, voc): os.path.join(audio_path, f'{participant}_{file_kind}_{voc}.wav') for condition, file_kind in [('oracle', 'ground_truth'), ('decoded', 'predicted')] for voc in vocoders}

    if len(complete_words) == 0 or not all(os.path.exists(path) for path in closeup_files.values()):
        print(f"Skipping close-up figure for {participant}: no complete word in the window or missing vocoder files")
    else:
        # use the first word whose display interval lies completely inside the plotted window
        word, word_start_s, word_end_s = complete_words[0]
        seg_start = int(round(word_start_s * rate))
        seg_len = int(round((word_end_s - word_start_s) * rate))
        orig_seg = orig_full[seg_start : seg_start + seg_len]

        # cut each vocoded file so that it covers the same stretch of the recording as orig_seg
        segments = {}
        for (condition, voc), path in closeup_files.items():
            voc_audio = load_wav_float(path, rate)

            # vocoded sample n matches original sample n + lag using the expected lag if this participant has no estimate
            expected_lag = expected_vocoder_lag(voc, trimmed_frames, hop_length, fftsize)
            lag = vocoder_lags.get(participant, {}).get(voc, expected_lag)
            segments[(condition, voc)] = voc_audio[seg_start - lag : seg_start - lag + seg_len]

        column_labels = ['Original recording', 'Griffin-Lim', 'HiFi-GAN']
        panel_letters = 'abcdef'

        fig, ax_close = plt.subplots(2, 3, sharex = True, sharey = True, figsize = (15, 7), constrained_layout = True)

        for row, condition in enumerate(conditions):
            # the original recording is repeated in both rows so that each row reads from left to right
            row_segments = [orig_seg, segments[(condition, 'griffinlim')], segments[(condition, 'hifigan')]]

            for col, (seg, col_label) in enumerate(zip(row_segments, column_labels)):
                ax = ax_close[row, col]
                seg_db = log_magnitude_db(seg, fftsize, overlap)

                # frame k starts at sample k * hop_length of the segment, as in the other spectrogram figures
                extent = [word_start_s, word_start_s + seg_db.shape[0] * hop_length / rate, 0, rate / 2]
                im = ax.imshow(seg_db.T, origin = 'lower', aspect = 'auto', cmap = 'viridis', vmin = -80, vmax = 0, extent = extent)

                ax.text(0.02, 0.95, col_label, transform = ax.transAxes, color = 'white', va = 'top', fontsize = 11)
                ax.set_title(panel_letters[3 * row + col], fontsize = 20)
            
            ax_close[row, 0].set_ylabel(f'{condition.capitalize()}\nFrequency (Hz)', fontsize = 13)
        
        # the first three formants and most of the harmonic energy lie below 4 kHz
        ax_close[0, 0].set_ylim(0, 4000)

        for ax in ax_close[1]:
            ax.set_xlabel('Time in recording (s)', fontsize = 13)

        # one colour bar serves all panels because they share the same colour range
        fig.colorbar(im, ax = ax_close, label = 'Level relative to panel maximum (dB)')
        fig.suptitle(f'{participant}, word "{word}"', fontsize = 14)

        plt.savefig(os.path.join(figures_path, 'vocoder_closeup.png'), dpi = 600)
        plt.show()