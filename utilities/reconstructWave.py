import time
import numpy as np
import matplotlib.pyplot as plt

def stft(x, fftsize = 1024, overlap = 4):
    """
    Compute the Short-Time Fourier Transform (STFT) of a 1D time-domain signal

    Parameters
    ----------
    x : array_like
        1D input signal (audio waveform)
    fftsize : int, default = 1024
        FFT window size in samples
    overlap : int, default = 4
        Overlap factor determining hop size (hop = fftsize / overlap)
    
    Returns
    -------
    res : ndarray
        Complex 2D spectrogram matrix of shape (time_frames, fftsize // 2 + 1)
    """
    # flatten input and cast to float to prevent 2D shape broadcasting issues
    x = np.asarray(x, dtype = float).ravel()

    # compute hop size in samples
    hop = int(fftsize / overlap)

    # ensure the signal is at least 1 full FFT window long
    if len(x) < fftsize:
        x = np.pad(x, (0, fftsize - len(x)))

    # construct periodic Hanning window
    w = np.hanning(fftsize + 1)[:-1]

    # generate frame start indices
    frames = range(0, len(x) - int(fftsize) + 1, hop)

    # compute real FFT across frames
    res = np.array([np.fft.rfft(w * x[i : i + fftsize]) for i in frames])

    return res

def istft(X, overlap = 4):
    """
    Compute the Inverse Short-Time Fourier Transform (iSTFT) of a 2D complex spectrum

    Parameters
    ----------
    X : ndarray
        Complex 2D array of shape (time_frames, frequency_bins)
    overlap : int, default = 4
        Overlap factor (e.g. 4 corresponds to 75% frame overlap)
    
    Returns
    -------
    x : ndarray
        1D reconstructed time-domain signal normalised by the summed squared window
        (weighted overlap-add) with the edge samples covered by too few frames attenuated
        rather than amplified
    """
    n_frames, spec_size = X.shape
    if n_frames == 0:
        return np.array([], dtype = float)

    # infer FFT size from number of positive frequency bins: fftsize = (bins - 1) * 2
    fftsize = (spec_size - 1) * 2
    hop = int(fftsize / overlap)

    # reconstruct periodic Hanning window
    w = np.hanning(fftsize + 1)[:-1]

    # calculate exact total output length required for all frames
    output_len = int((n_frames - 1) * hop + fftsize)
    x = np.zeros(output_len, dtype = np.float64)
    wsum = np.zeros(output_len, dtype = np.float64)

    # loop directly over all frames in x (ensuring no frames are dropped)
    for n in range(n_frames):
        sample_idx = n * hop

        # compute inverse real FFT for frame n
        frame_time = np.real(np.fft.irfft(X[n], n = fftsize)) * w

        # overlap-add into output arrays
        x[sample_idx : sample_idx + fftsize] += frame_time
        wsum[sample_idx : sample_idx + fftsize] += w ** 2.0

    # normalise by the summed squared window with a floor of 10% of its maximum so that the edges are not amplified
    x /= np.maximum(wsum, 0.1 * np.max(wsum))

    return x

def reconstructWavFromSpectrogram(spec, lenWaveFile, fftsize = 1024, overlap = 4, numIterations = 32, rng = None):
    """
    Reconstruct a time-domain audio waveform from a magnitude spectrogram
    using the Griffin-Lim phase retrieval algorithm (Griffin and Lim, 1984)

    Parameters
    ----------
    spec : ndarray
        Magnitude spectrogram matrix of shape (time_frames, frequency_bins)
    lenWaveFile : int
        Target output signal length in audio samples
    fftsize : int, default = 1024
        FFT window size in samples
    overlap : int, default = 4
        Window overlap factor
    numIterations : int, default = 32
        Number of Griffin-Lim phase retrieval iterations
    rng : np.random.Generator, optional
        Random number generator used for the Gaussian noise initialisation
    
    Returns
    -------
    reconstructedWav : ndarray
        1D reconstructed time-domain audio wave truncated to lenWaveFile samples
    """
    if rng is None:
        rng = np.random.default_rng()

    # ensure working buffer is at least 1 fftsize long so STFT has valid dimensions
    buffer_len = max(lenWaveFile, fftsize)

    # initialise with zero-mean Gaussian noise (prevents positive DC bias)
    reconstructedWav = rng.standard_normal(buffer_len)

    for i in range(numIterations):
        # compute STFT of current time-domain estimate
        x = stft(reconstructedWav, fftsize = fftsize, overlap = overlap)

        # determine maximum common time-frame length to prevent dimension mismatch
        n_frames = min(spec.shape[0], x.shape[0])
        if n_frames == 0:
            break

        # combine target magnitude with estimated phase angle: z = |spec| * exp(j * angle(x))
        angles = np.angle(x[:n_frames, :])
        z = spec[:n_frames, :] * np.exp(1j * angles)

        # project back to the time domain with the weighted overlap-add iSTFT
        re = istft(z, overlap = overlap)

        # update wave buffer safely
        if len(re) >= buffer_len:
            reconstructedWav = re[:buffer_len]
        else:
            reconstructedWav = np.zeros(buffer_len)
            reconstructedWav[:len(re)] = re
    
    # final safety truncation to target sample length
    return reconstructedWav[:lenWaveFile]

if __name__ == "__main__":
    sr = 16000
    duration = 3.0
    print(f"Generating a {duration}-second synthetic test tone (440 Hz + 880 Hz) at {sr} Hz...")

    t_arr = np.linspace(0, duration, int(sr * duration), endpoint = False)
    wf = 0.5 * np.sin(2 * np.pi * 440 * t_arr) + 0.25 * np.sin(2 * np.pi * 880 * t_arr)

    # compute Forward STFT and magnitude spectrogram
    x = stft(wf, fftsize = 1024, overlap = 4)
    spectrogram = np.abs(x)
    print(f"STFT Matrix Shape: {x.shape} (Time Frames x Frequency Bins)")

    # plot spectrogram
    plt.figure(figsize = (10, 4))
    plt.imshow(np.log(spectrogram + 1e-8).T, origin = 'lower', aspect = 'auto', cmap = 'viridis')
    plt.title("Log-Magnitude Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("FFT Frequency Bins")
    plt.colorbar(label = "Log Amplitude")
    plt.show()

    # perform Griffin-Lim audio reconstruction
    print("Running Griffin-Lim phase retrieval (32 iterations)...")
    start_time = time.time()
    reconstructed_data = reconstructWavFromSpectrogram(
        spectrogram,
        lenWaveFile = len(wf),
        fftsize = 1024,
        overlap = 4,
        numIterations = 32
    )
    elapsed = time.time() - start_time
    print(f"Reconstruction completed in {elapsed:.3f} seconds.")