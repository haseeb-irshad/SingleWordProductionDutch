import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal
from pynwb import NWBHDF5IO
from dataclasses import dataclass

@dataclass # automatically generates the entire __init__ function behind the scenes
class RawParticipantData:
    """
    A standardised container holding raw, unprocessed array data and metadata for a
    single participant extracted directly from iBIDS NWB files

    Attributes
    ----------
    participant_id : str
        Subject identifier string (e.g. 'sub-01')
    eeg : np.ndarray
        2D numerical array containing raw multichannel stereotactic EEG (sEEG)
        voltage time series data formatted with shape (samples, channels)
    eeg_sr : float
        Sampling rate of the sEEG recordings in Hz (downsampled to 1024.0 Hz)
    audio : np.ndarray
        1D numerical array containing raw 1-channel speech audio waveform data
        formatted with shape (samples,). Audio is pitch-shifted for anonymity
    audio_sr : float
        Sampling rate of the recorded microphone audio in Hz (recorded at 48000.0 Hz)
    words : np.ndarray
        1D categorical array containing string labels for presented Dutch stimulus words
        (IFA corpus), formatted with shape (samples,) and aligned to the eeg sampling
        rate (eeg_sr)
    channels : pd.DataFrame
        Pandas DataFrame parsed from the BIDS '_channels.tsv' file specifying
        channel names, signal types, measurement units, low/high cutoff frequencies,
        anatomical region descriptions, sampling frequencies and channel status
    electrodes : pd.DataFrame
        Pandas DataFrame parsed from the BIDS '_electrodes.tsv' file specifying
        columns 'name', 'x', 'y', 'z' (3D coordinates in ACPC, anterior commissure-
        posterior commissure, stereotactic space, in mm) and 'size'
    """
    participant_id: str
    eeg: np.ndarray
    eeg_sr: float
    audio: np.ndarray
    audio_sr: float
    words: np.ndarray
    channels: pd.DataFrame
    electrodes: pd.DataFrame

class BIDSDataLoader:
    """
    Handles data ingestion, BIDS directory validation and PyNWB array loading
    for iEEG datasets

    Attributes
    ----------
    bids_root : str
        Absolute filesystem path pointing to the root directory of the iBIDS dataset
    participants_tsv_path : str
        Absolute path to the master 'participants.tsv' metadata file
    """

    def __init__(self, bids_root):
        """
        Initialise the BIDSDataLoader instance and validate the dataset folder path

        Parameters
        ----------
        bids_root : str
            Relative or absolute path to the iBIDS root dataset directory

        Raises
        ------
        FileNotFoundError
            If the specified `bids_root` directory does not exist on the disk
        """
        self.bids_root = os.path.abspath(bids_root)
        if not os.path.exists(self.bids_root):
            raise FileNotFoundError(f"BIDS root directory not found: {self.bids_root}")
        self.participants_tsv_path = os.path.join(self.bids_root, 'participants.tsv')

    def load_participants_table(self):
        """
        Parse the master participants.tsv metadata file

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame containing participant demographic records,
            task parameters and participant IDs (e.g. 'sub-01')

        Raises
        ------
        FileNotFoundError
            If 'participants.tsv' is missing from the dataset root folder
        """
        if not os.path.exists(self.participants_tsv_path):
            raise FileNotFoundError(f"participants.tsv missing at: {self.participants_tsv_path}")
        return pd.read_csv(self.participants_tsv_path, delimiter = '\t')

    def load_channel_metadata(self, participant_id):
        """
        Parse the subject-specific _channels.tsv electrode metadata file

        Parameters
        ----------
        participant_id : str
            Unique participant identifier string (e.g. 'sub-01')

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame containing channel names, signal types, measurement
            units, low/high cutoff frequencies, anatomical descriptions,
            sampling frequencies and channel status flags

        Raises
        ------
        FileNotFoundError
            If the subject's channel metadata TSV file cannot be found
        """
        channels_path = os.path.join(self.bids_root, participant_id, 'ieeg', f'{participant_id}_task-wordProduction_channels.tsv')
        if not os.path.exists(channels_path):
            raise FileNotFoundError(f"Channels TSV not found at: {channels_path}")
        return pd.read_csv(channels_path, delimiter = '\t')

    def load_electrode_coordinates(self, participant_id):
        """
        Parse the subject-specific _electrodes.tsv file containing each
        electrode's 3D stereotactic coordinates

        Parameters
        ----------
        participant_id : str
            Unique participant identifier string (e.g. 'sub-01')

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with columns 'name', 'x', 'y', 'z' (3D coordinates in ACPC
            stereotactic space, in mm) and 'size'

        Raises
        ------
        FileNotFoundError
            If the subject's ACPC-space electrode coordinates TSV file cannot be found
        """
        electrodes_path = os.path.join(self.bids_root, participant_id, 'ieeg', f'{participant_id}_task-wordProduction_space-ACPC_electrodes.tsv')
        if not os.path.exists(electrodes_path):
            raise FileNotFoundError(f"Electrodes TSV not found at: {electrodes_path}")
        return pd.read_csv(electrodes_path, delimiter = '\t')

    def load_raw_participant(self, participant_id):
        """
        Safely open an NWB HDF5 container, extract raw signal arrays fully into
        memory and package them into a structured RawParticipantData container

        Parameters
        ----------
        participant_id : str
            Unique participant identifier string (e.g. 'sub-01')

        Returns
        -------
        RawParticipantData
            Dataclass instance holding raw multichannel iEEG arrays, single-channel
            audio waveforms, stimulus word arrays, hardware sampling rates, tabular
            electrode channel specifications and 3D electrode coordinates

        Raises
        ------
        FileNotFoundError
            If the subject's `.nwb` binary acquisition file is missing from the disk
            or if the subject's `_channels.tsv` metadata file is missing (raised by
            the internal call to `load_channel_metadata`) or if the subject's
            `_space-ACPC_electrodes.tsv` file is missing (raised by the internal call
            to `load_electrode_coordinates`)
        ValueError
            If the sampling frequencies listed in the subject's `_channels.tsv` file are
            not all equal to the assumed iEEG sampling rate (`eeg_sr`)
        """
        nwb_path = os.path.join(self.bids_root, participant_id, 'ieeg', f'{participant_id}_task-wordProduction_ieeg.nwb')
        if not os.path.exists(nwb_path):
            raise FileNotFoundError(f"NWB file missing at: {nwb_path}")

        # open NWB file safely using a manual try/finally block, guaranteeing io.close()
        io = NWBHDF5IO(nwb_path, 'r')
        try:
            nwbfile = io.read()
            # load raw array data fully into memory
            # the [:] slice forces the full array to be read from the disk into RAM
            eeg = nwbfile.acquisition['iEEG'].data[:]
            audio = nwbfile.acquisition['Audio'].data[:]
            words = nwbfile.acquisition['Stimulus'].data[:]
            words = np.array(words, dtype = str)

            # the dataset description states that sEEG recorded at 1024 or 2048 Hz was downsampled to 1024 Hz
            # audio was recorded at 48 kHz with the onboard microphone and pitch-shifted for anonymity
            eeg_sr = 1024.0
            audio_sr = 48000.0
        finally:
            io.close()

        # load tabular channel metadata and 3D electrode coordinates
        channels_df = self.load_channel_metadata(participant_id)
        electrodes_df = self.load_electrode_coordinates(participant_id)

        # cross-check the hardcoded sampling rate against the BIDS channel metadata
        sr_meta = channels_df['sampling_frequency'].astype(float).unique()
        if len(sr_meta) != 1 or sr_meta[0] != eeg_sr:
            raise ValueError(f"{participant_id}: _channels.tsv sampling rates {sr_meta} != assumed {eeg_sr} Hz")

        return RawParticipantData(
            participant_id = participant_id,
            eeg = eeg,
            eeg_sr = eeg_sr,
            audio = audio,
            audio_sr = audio_sr,
            words = words,
            channels = channels_df,
            electrodes = electrodes_df
        )

def validate_raw_data(raw_data):
    """
    Perform pre-processing signal integrity and quality assurance checks on raw participant data

    Evaluates array validity by scanning stereotactic EEG (sEEG) and microphone audio arrays
    for numerical anomalies (NaN or Inf values) and identifies dead or unresponsive channels
    displaying near-zero signal variance across time

    Parameters
    ----------
    raw_data : RawParticipantData
        Standardised container holding raw participant time-series arrays, hardware sampling
        rates and electrode specifications

    Returns
    -------
    dict
        Dictionary containing quantitative signal health metrics with the following keys:
        - 'participant_id' (str): Subject identifier string (e.g. 'sub-01')
        - 'has_nans_or_infs' (bool): True if any NaN or Inf values exist in sEEG or audio arrays
        - 'dead_channels' (list of int): List of channel indices that are either near-zero
          variance (standard deviation < 1e-6) or entirely invalid (100% NaN/Inf). Note this is a
          separate check from 'has_nans_or_infs': a channel can be partially NaN without
          appearing here and a channel can appear here without any NaN/Inf at all
        - 'num_dead_channels' (int): Total count of detected dead or unresponsive sEEG channels
    """
    has_invalid = (
        np.isnan(raw_data.eeg).any() or
        np.isinf(raw_data.eeg).any() or
        np.isnan(raw_data.audio).any() or
        np.isinf(raw_data.audio).any()
    )

    # identify dead channels (near-zero variance)
    eeg_filled = np.where(np.isfinite(raw_data.eeg), raw_data.eeg, 0.0)
    ch_stds = np.std(eeg_filled, axis = 0)
    dead_channels = np.where(ch_stds < 1e-6)[0].tolist()

    return {
        'participant_id': raw_data.participant_id,
        'has_nans_or_infs': has_invalid,
        'dead_channels': dead_channels,
        'num_dead_channels': len(dead_channels),
    }

def compute_low_freq_power_ratio(frequencies, psd, cutoff = 1.0):
    """
    Calculate the proportion of total spectral power residing below a low-frequency cutoff

    This metric quantifies low-frequency baseline drifts (e.g. 0-1 Hz DC offsets)
    in iEEG signals by integrating the Welch Power Spectral Density (PSD) estimate
    up to the cutoff frequency and normalising by total signal power

    Parameters
    ----------
    frequencies : np.ndarray
        1D array of frequency bin centres in Hz output by a PSD estimator (e.g. Welch)
    psd : np.ndarray
        1D array of power spectral density estimates corresponding to each frequency bin
    cutoff : float, default = 1.0
        Upper frequency boundary in Hz for the low-frequency band of interest

    Returns
    -------
    float
        Fractional power ratio (ranging from 0.0 to 1.0) contained below `cutoff` Hz
        Returns 0.0 if the input frequency array contains fewer than two points
    """
    if len(frequencies) < 2:
        return 0.0

    df = frequencies[1] - frequencies[0]
    mask = frequencies <= cutoff

    # approximate power as area under the PSD curve
    p_low = np.sum(psd[mask]) * df
    p_tot = np.sum(psd) * df
    return float(p_low / (p_tot + 1e-16))

def inspect_channel_detrending(raw_data, channel_idx = None, save_fig_path = None):
    """
    Perform exploratory signal analysis on a single iEEG electrode channel to
    evaluate the effect of linear detrending on low-frequency baseline drift

    This function isolates a specific channel (or automatically selects the channel
    with the highest overall variance), applies linear detrending, computes the Welch
    Power Spectral Density (PSD) estimate and the low-frequency (0-1 Hz) power ratio
    for both the raw and detrended signal and visualises the time series waveform,
    amplitude distribution and PSD for each side by side. Raw and detrended traces
    are plotted in separate columns rather than overlaid, since overlapping traces
    are difficult to tell apart by eye

    Parameters
    ----------
    raw_data : RawParticipantData
        Standardised data container holding raw participant arrays, sampling rates
        and channel specifications
    channel_idx : int, optional
        Zero-based index of the target iEEG channel to inspect. If None, defaults to
        the channel displaying the maximum standard deviation across time
    save_fig_path : str, optional
        File path where the generated six-panel diagnostic figure will be saved
        (e.g. 'results/sub-01_detrend_check.png'). If None, the figure is rendered
        interactively without saving to the disk

    Returns
    -------
    channel_idx : int
        Zero-based index of the evaluated target iEEG channel
    ratio_raw : float
        Proportion of the total spectral power residing below 1.0 Hz in the raw signal
    ratio_detrended : float
        Proportion of the total spectral power residing below 1.0 Hz in the detrended signal
    """
    eeg = raw_data.eeg
    sr = raw_data.eeg_sr

    # select channel with maximum variance if channel_idx is not explicitly passed
    if channel_idx is None:
        channel_idx = int(np.nanargmax(np.nanstd(eeg, axis = 0)))

    x = eeg[:, channel_idx].astype(float)
    n = len(x)
    t = np.arange(n) / sr

    # apply linear detrending to remove slow drifts (e.g. electrode drift)
    x_detrended = scipy.signal.detrend(x)

    # Welch PSD computation
    nperseg = min(65536, max(1024, n // 8))
    f_raw, P_raw = scipy.signal.welch(x, fs = sr, nperseg = nperseg)
    f_detrended, P_detrended = scipy.signal.welch(x_detrended, fs = sr, nperseg = nperseg)

    ratio_raw = compute_low_freq_power_ratio(f_raw, P_raw)
    ratio_detrended = compute_low_freq_power_ratio(f_detrended, P_detrended)

    # plots: 3 rows (time series, histogram, PSD) x 2 columns (raw, detrended)
    fig = plt.figure(figsize = (10, 7))

    # time-series plots
    plt.subplot(3, 2, 1)
    plt.plot(t, x, color = 'blue')
    plt.xlabel('Time / s', fontsize = 10)
    plt.ylabel('Amplitude / µV', fontsize = 10)
    plt.title(f'Subject {raw_data.participant_id} | Channel {channel_idx}: Raw Signal', fontsize = 12)

    plt.subplot(3, 2, 2)
    plt.plot(t, x_detrended, color = 'red')
    plt.xlabel('Time / s', fontsize = 10)
    plt.ylabel('Amplitude / µV', fontsize = 10)
    plt.title(f'Subject {raw_data.participant_id} | Channel {channel_idx}: Detrended Signal', fontsize = 12)

    # histogram plots
    plt.subplot(3, 2, 3)
    plt.hist(x, bins = 200, label = f'Raw (0-1 Hz Power Ratio = {ratio_raw:.3f})', color = 'blue')
    plt.xlabel('Amplitude / µV', fontsize = 10)
    plt.ylabel('Count', fontsize = 10)
    plt.title('Amplitude Distribution (Raw)', fontsize = 12)
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.hist(x_detrended, bins = 200, label = f'Detrended (0-1 Hz Power Ratio = {ratio_detrended:.3f})', color = 'red')
    plt.xlabel('Amplitude / µV', fontsize = 10)
    plt.ylabel('Count', fontsize = 10)
    plt.title('Amplitude Distribution (Detrended)', fontsize = 12)
    plt.legend()

    # Welch PSD plots
    plt.subplot(3, 2, 5)
    plt.semilogy(f_raw, P_raw, label = 'Raw PSD', color = 'blue')
    plt.xlim(0, min(200, sr / 2))
    plt.xlabel('Frequency / Hz', fontsize = 10)
    plt.ylabel('PSD / µV^2 Hz^-1', fontsize = 10)
    plt.title('Power Spectral Density, Raw (Welch)', fontsize = 12)
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.semilogy(f_detrended, P_detrended, label = 'Detrended PSD', color = 'red')
    plt.xlim(0, min(200, sr / 2))
    plt.xlabel('Frequency / Hz', fontsize = 10)
    plt.ylabel('PSD / µV^2 Hz^-1', fontsize = 10)
    plt.title('Power Spectral Density, Detrended (Welch)', fontsize = 12)
    plt.legend()

    plt.tight_layout()

    if save_fig_path:
        plt.savefig(save_fig_path, dpi = 300)
    plt.show()
    plt.close(fig)

    return channel_idx, ratio_raw, ratio_detrended

if __name__ == "__main__":
    # resolve directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bids_directory = os.path.join(script_dir, 'SingleWordProductionDutch-iBIDS')

    # initialise loader
    loader = BIDSDataLoader(bids_root = bids_directory)

    # inspect participants metadata with Pandas
    print("Parsing Dataset Participants Table")
    participants_df = loader.load_participants_table()
    print(participants_df.to_string(index = False))

    # iterate through participants, load raw containers and execute checks
    raw_dataset_summary = []

    for participant_id in participants_df['participant_id']:
        print(f"Loading raw BIDS data for: {participant_id}...")
        raw_data = loader.load_raw_participant(participant_id)

        # verify raw integrity
        diagnostics = validate_raw_data(raw_data)
        if diagnostics['has_nans_or_infs']:
            print(f"Corrupt raw data detected in {participant_id} (NaNs/Infs present)")
        if diagnostics['num_dead_channels'] > 0:
            print(f"{diagnostics['num_dead_channels']} dead channel(s) detected in {participant_id}: indices {diagnostics['dead_channels']}")

        num_channels_metadata = len(raw_data.channels)
        num_channels_eeg = raw_data.eeg.shape[1]
        channel_count_mismatch = num_channels_metadata != num_channels_eeg
        if channel_count_mismatch:
            print(f"Channel count mismatch in {participant_id}: "
                  f"_channels.tsv has {num_channels_metadata} rows, "
                  f"eeg array has {num_channels_eeg} channels")

        # basic array properties
        eeg_duration_sec = raw_data.eeg.shape[0] / raw_data.eeg_sr
        audio_duration_sec = raw_data.audio.shape[0] / raw_data.audio_sr

        summary = {
            'participant': participant_id,
            'eeg_shape': str(raw_data.eeg.shape),
            'eeg_sr_hz': raw_data.eeg_sr,
            'eeg_duration_s': round(eeg_duration_sec, 2),
            'audio_samples': raw_data.audio.shape[0],
            'audio_sr_hz': raw_data.audio_sr,
            'audio_duration_s': round(audio_duration_sec, 2),
            'num_channels': num_channels_metadata,
            'channel_count_mismatch': channel_count_mismatch,
            'has_nans_or_infs': diagnostics['has_nans_or_infs'],
            'num_dead_channels': diagnostics['num_dead_channels'],
            'dead_channel_indices': diagnostics['dead_channels']
        }
        raw_dataset_summary.append(summary)

        # run exploratory detrending analysis on sub-01
        if participant_id == 'sub-01':
            print("\nChannels DataFrame Inspection (sub-01)")
            print(f"Shape: {raw_data.channels.shape} (rows, columns)")
            print(f"Columns: {raw_data.channels.columns.tolist()}")
            print("First 10 rows:")
            print(raw_data.channels.head(10).to_string())

            print("\nElectrodes DataFrame Inspection (sub-01)")
            print(f"Shape: {raw_data.electrodes.shape} (rows, columns)")
            print(f"Columns: {raw_data.electrodes.columns.tolist()}")
            print("First 10 rows:")
            print(raw_data.electrodes.head(10).to_string())

            # check for unique 'size' values in the electrodes DataFrame
            print(f"Unique 'size' values: {raw_data.electrodes['size'].unique().tolist()}")

            print("\nWords Stimulus Array Inspection (sub-01)")
            print(f"Total Sample Length: {raw_data.words.shape[0]}")
            print(f"Data Type: {raw_data.words.dtype}")

            # calculate frequency of unique words/tokens across the recording
            unique_tokens, counts = np.unique(raw_data.words, return_counts = True)
            words_freq = pd.DataFrame({'token': unique_tokens, 'sample_count': counts}).sort_values(by = 'sample_count', ascending = False)
            
            print("Top 15 Unique Stimulus Tokens (by sample duration):")
            print(words_freq.head(15).to_string(index = False))

            print(f"\nExecuting baseline signal analysis for {participant_id}...")
            channel_idx, ratio_raw, ratio_detrended = inspect_channel_detrending(raw_data)
            print(f"Channel {channel_idx} low-freq ratio (0-1 Hz): raw = {ratio_raw:.4f}, detrended = {ratio_detrended:.4f}")

    # display overall raw dataset summary table
    summary_df = pd.DataFrame(raw_dataset_summary)
    print("Raw iBIDS Dataset Structural Summary")
    print(summary_df.to_string(index = False))