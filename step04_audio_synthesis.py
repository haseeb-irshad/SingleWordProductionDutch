import os
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.utils.fetching import LocalStrategy
from utilities import reconstructWave as rW
from utilities import MelFilterBank as mel

def synthesise_Griffin_Lim(spectrogram, audiosr, windowLength, frameshift, seed = 0):
    """
    Reconstruct a time-domain audio waveform from an un-normalised log-Mel spectrogram using Griffin-Lim synthesis

    Parameters
    ----------
    spectrogram : np.ndarray
        Un-normalised log-Mel spectrogram matrix with shape (windows, mel_bins)
    audiosr : int
        Sampling rate of the target audio signal in Hz (e.g. 16000 Hz)
    windowLength : float
        Length of window (in seconds) over which spectrogram frames were computed
    frameshift : float
        Shift (in seconds) between consecutive sliding windows
    seed : int, default = 0
        Random seed for reproducibility of Griffin-Lim phase estimation
    
    Returns
    -------
    scaled : np.ndarray
        Peak-normalised 16-bit PCM reconstructed audio waveform array formatted as int16
    """
    # compute FFT window size in samples (e.g. 16000 Hz * 0.064 s = 1024 samples)
    fft_size = int(audiosr * windowLength)

    # initialise Mel Filterbank matching FFT frequency bins (1024//2 + 1 = 513 bins)
    mfb = mel.MelFilterBank(int(fft_size / 2 + 1), spectrogram.shape[1], audiosr)

    # invert log-Mel spectrum back to linear FFT magnitude spectrum (shape: time_frames * 513)
    for_reconstruction = mfb.fromLogMels(spectrogram)

    # calculate exact time-domain sample length (0.016 * 16000 = 256 samples)
    samples_per_shift = int(frameshift * audiosr)
    lenWaveFile = int((spectrogram.shape[0] - 1) * samples_per_shift + fft_size)

    # synthesise continuous waveform via Griffin-Lim phase estimation
    rec_audio = rW.reconstructWavFromSpectrogram(
        for_reconstruction,
        lenWaveFile = lenWaveFile,
        fftsize = fft_size,
        overlap = int(windowLength / frameshift),
        rng = np.random.default_rng(seed)
    )

    # peak-normalise to 16-bit integer PCM range [-32767, 32767]
    max_val = np.max(np.abs(rec_audio))
    if max_val > 0:
        scaled = np.round(rec_audio / max_val * 32767).astype(np.int16)
    else:
        scaled = rec_audio.astype(np.int16)

    return scaled

def load_hifigan_vocoder(source = "speechbrain/tts-hifigan-libritts-16kHz", run_opts = None):
    """
    Load a pretrained SpeechBrain HiFi-GAN vocoder model

    Parameters
    ----------
    source : str, default = "speechbrain/tts-hifigan-libritts-16kHz"
        Hugging Face Hub identifier of the SpeechBrain HiFi-GAN checkpoint to load. The
        default is the 16 kHz LibriTTS checkpoint used elsewhere in this project
    run_opts : dict, optional
        Runtime execution options dictionary passed to SpeechBrain (e.g. device assignment)

    Returns
    -------
    hifi_gan : HIFIGAN
        Initialised SpeechBrain HiFi-GAN neural vocoder instance
    """
    # build a local cache folder named after the chosen checkpoint
    checkpoint_name = source.split("/")[-1]
    savedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models", checkpoint_name)
    hifi_gan = HIFIGAN.from_hparams(
        source = source,
        savedir = savedir,
        run_opts = run_opts,
        local_strategy = LocalStrategy.COPY, # prevents Windows symlink privilege error
    )
    print(f"HiFi-GAN Model Loaded: {source}")
    return hifi_gan

def synthesise_audio_waveform(spec_zscored, mel_mean, mel_std, vocoder_model):
    """
    Synthesise a time-domain audio waveform from a Z-scored log-Mel spectrogram using HiFi-GAN

    Parameters
    ----------
    spec_zscored : np.ndarray
        Z-normalised log-Mel spectrogram matrix formatted with shape (frames, mel_bins)
    mel_mean : np.ndarray or float
        Mean value(s) used to un-scale features back to natural log-Mel magnitudes
    mel_std : np.ndarray or float
        Standard deviation value(s) used to un-scale features
    vocoder_model : HIFIGAN
        Pretrained SpeechBrain HiFi-GAN neural vocoder model instance

    Returns
    -------
    audio_pcm16 : np.ndarray
        Peak-normalised 16-bit PCM reconstructed audio waveform array formatted as int16
    """
    # undo the Z-scoring to return to the natural log-Mel scale
    spec_unscaled = (spec_zscored * mel_std) + mel_mean

    # transpose matrix from [T, 80] to [80, T] and add batch dimension [1, 80, T]
    mel_tensor = torch.from_numpy(spec_unscaled).T.unsqueeze(0).float()

    # ensure tensor moves to device specified by the vocoder model
    device = next(vocoder_model.parameters()).device
    mel_tensor = mel_tensor.to(device)

    # generate time-domain waveform using SpeechBrain HiFi-GAN decode_batch
    with torch.no_grad():
        waveform_tensor = vocoder_model.decode_batch(mel_tensor)

    # convert PyTorch tensor output to a 1D NumPy float32 array
    waveform_np = waveform_tensor.squeeze().cpu().numpy()

    # normalise peak audio amplitude to range [-1.0, 1.0] to prevent digital clipping
    max_val = np.max(np.abs(waveform_np))
    if max_val > 0:
        waveform_np = waveform_np / max_val

    # convert float32 [-1.0, 1.0] to 16-bit PCM integer values
    audio_pcm16 = np.round(waveform_np * 32767).astype(np.int16)

    return audio_pcm16

if __name__ == "__main__":
    # resolve execution directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_save_dir = os.path.join(script_dir, 'results', 'audio')
    os.makedirs(audio_save_dir, exist_ok = True)

    # audio STFT signal reconstruction parameters
    windowLength = 0.064
    frameshift = 0.016
    audiosr = 16000

    # load the pretrained HiFi-GAN neural vocoder once before the loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocoder_model = load_hifigan_vocoder(run_opts = {"device": device})

    # execute synthesis pipeline across all participants
    for i in range(1, 11):
        participant_id = f'sub-{i:02d}'
        print(f"\nProcessing {participant_id}...")

        gt_path = os.path.join(script_dir, 'features', f'{participant_id}_spec.npy')
        pred_path = os.path.join(script_dir, 'results', f'{participant_id}_predicted_spec.npy')

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"Skipping {participant_id}: missing ground-truth or predicted spectrogram file.")
            continue

        gt_spec = np.load(gt_path)
        pred_spec = np.load(pred_path)

        if gt_spec.shape != pred_spec.shape:
            print(f"Skipping {participant_id}: shape mismatch GT {gt_spec.shape} vs Pred {pred_spec.shape}")
            continue

        # load normalisation parameters
        norm_path = os.path.join(script_dir, 'features', f'{participant_id}_mel_norm.npy')

        if not os.path.exists(norm_path):
            print(f"Skipping {participant_id}: missing normalisation file")
            continue

        mel_norm = np.load(norm_path)
        mel_mean, mel_std = mel_norm[0], mel_norm[1]

        # compute un-scaled spectrograms for Griffin-Lim
        gt_spec_unnorm = (gt_spec * mel_std) + mel_mean
        pred_spec_unnorm = (pred_spec * mel_std) + mel_mean

        # Griffin-Lim audio synthesis
        print("Synthesising Griffin-Lim Audio...")
        gl_gt_audio = synthesise_Griffin_Lim(gt_spec_unnorm, audiosr, windowLength, frameshift)
        gl_pred_audio = synthesise_Griffin_Lim(pred_spec_unnorm, audiosr, windowLength, frameshift)
        wavfile.write(os.path.join(audio_save_dir, f'{participant_id}_ground_truth_griffinlim.wav'), audiosr, gl_gt_audio)
        wavfile.write(os.path.join(audio_save_dir, f'{participant_id}_predicted_griffinlim.wav'), audiosr, gl_pred_audio)

        # HiFi-GAN neural audio synthesis
        print("Synthesising HiFi-GAN Neural Audio...")
        hifi_gt_audio = synthesise_audio_waveform(gt_spec, mel_mean, mel_std, vocoder_model)
        hifi_pred_audio = synthesise_audio_waveform(pred_spec, mel_mean, mel_std, vocoder_model)
        wavfile.write(os.path.join(audio_save_dir, f'{participant_id}_ground_truth_hifigan.wav'), audiosr, hifi_gt_audio)
        wavfile.write(os.path.join(audio_save_dir, f'{participant_id}_predicted_hifigan.wav'), audiosr, hifi_pred_audio)