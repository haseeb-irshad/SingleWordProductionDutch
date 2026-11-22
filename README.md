# Speech Decoding from sEEG with Griffin-Lim and HiFi-GAN Synthesis

I worked on this project during a summer research internship at the Advanced Biomedical Instrumentation Centre in Hong Kong in 2026. It builds on the code that accompanies the *Dataset of Speech Production in intracranial Electroencephalography* (Verwoert et al., 2022). In that study, ten native Dutch speakers with drug-resistant epilepsy each read 100 words aloud from a laptop screen. Their brain activity was recorded with stereo-electroencephalography (sEEG), a form of intracranial EEG, using electrodes implanted as part of their clinical treatment. The pipeline reconstructs each participant's speech spectrogram from their brain activity alone and turns the result back into audio.

The original code by the dataset authors is available at [neuralinterfacinglab/SingleWordProductionDutch](https://github.com/neuralinterfacinglab/SingleWordProductionDutch).

## Why I made these changes

The original authors turned the decoded spectrograms back into audio with Griffin-Lim. They found that the timing of the reconstructed speech matched the original closely, but that much of the audio quality was lost in this step. They suggested trying a neural vocoder such as WaveGlow or WaveNet. I wanted to test this with HiFi-GAN, a more recent neural vocoder. To compare the two fairly, I first had to make sure that the neural features, the spectrograms and the audio were all aligned in time. I also needed a way to separate the loss caused by the vocoder from the loss caused by the decoder.

## Changes from the original code

### Features and decoding
- Spectrograms use 80 Mel bins on the Slaney scale with 64 ms windows (1,024 samples) and a 16 ms step (256 samples) at 16 kHz. They are computed with SpeechBrain's `mel_spectrogram` using the settings that the HiFi-GAN checkpoint was trained on. The original code used 23 Mel bins with 50 ms windows and a 10 ms step.
- The high-gamma features use the same 64 ms windows and 16 ms step, so each feature window lines up with one spectrogram frame. The filtering is the same as before (70-170 Hz band-pass, band-stop filters at 100 and 150 Hz, Hilbert envelope). I also added a common median reference, computed from the channels that are neither flat nor unusually noisy, and a log transform of the envelope before averaging.
- Temporal context comes from stacking four windows before and four after the current one, spaced 48 ms (3 frames) apart. This covers ±192 ms, close to the original ±200 ms.
- torchaudio centres each STFT frame by padding 512 samples (32 ms) at the start of the audio, so each spectrogram frame started 32 ms before the matching sEEG window. I drop the first two frames so that frame *k* of both now starts at *k* × 16 ms.
- The decoder is unchanged. Features are z-scored with training-fold statistics, reduced to 50 principal components and mapped to the spectrogram with ordinary least squares, in 10-fold cross-validation without shuffling.

### Audio synthesis
- I added a pretrained HiFi-GAN vocoder (Kong, Kim and Bae, 2020) from SpeechBrain (Ravanelli et al., 2021), trained on LibriTTS (Zen et al., 2019) at 16 kHz, alongside Griffin-Lim (Griffin and Lim, 1984).
- The Mel filterbank that Griffin-Lim uses to invert the spectrogram now uses the Slaney Mel scale to match SpeechBrain. The original used the HTK formula.
- Griffin-Lim now runs for 32 iterations instead of 8.
- The inverse STFT now divides by the summed squared window, which the original computed but never applied, and keeps the last few frames, which the original dropped. The divisor is floored at 10% of its maximum because the samples at either end are covered by only one or two windows and would otherwise be amplified into a spike.

### Evaluation
- **Chance level.** The random control is the same as in the original paper. For each of 1,000 shifts, the spectrogram is cut at a random point at least 10% from either end, the two parts are swapped (a circular shift) and the result is correlated with the unshifted spectrogram. I added a p-value, (*b* + 1) / 1,001, where *b* is the number of shifts whose mean correlation across Mel bins matches or beats the decoder's (Phipson and Smyth, 2010).
- **Oracle and decoded conditions.** Each vocoder is run on the ground-truth spectrogram (oracle) and on the predicted spectrogram (decoded). The oracle condition measures the loss from the vocoder alone and the gap between the two conditions measures the loss from decoding.
- **Alignment.** The first 12 frames (192 ms) are lost when context is stacked, so the vocoded audio begins 192 ms into the recording for Griffin-Lim. For HiFi-GAN, the offset is about 144 ms. I refine each offset by cross-correlating the amplitude envelopes of the recording and the oracle audio, then apply it to the decoded audio before scoring.
- **Metrics.** Short-time objective intelligibility (STOI; Taal et al., 2011) is computed on the whole aligned recording. Log-spectral distance (LSD) compares STFT magnitudes after scaling the vocoded audio to the same overall energy as the original. Mel-cepstral distance (MCD) uses cepstral coefficients 1-15, leaving out c0, which only reflects loudness. LSD and MCD are averaged over speech frames only, meaning frames of the original recording within 40 dB of its loudest frame.
- **Statistics.** The vocoders are compared with two-sided Wilcoxon signed-rank tests across the ten participants with Holm-Bonferroni correction (Holm, 1979) across the six tests (three metrics in two conditions).
- **Electrode coverage.** The number of contacts in the superior temporal gyrus (STG) and ventral sensorimotor cortex (vSMC) is counted for each participant from the Destrieux atlas labels (Destrieux et al., 2010) and correlated with decoding performance (Pearson's *r*).

### Code structure
- I reorganised the original scripts (`extract_features.py`, `reconstruction_minimal.py` and `viz_results.py`) into five numbered scripts with a shared data loader and NumPy-style docstrings (see [Running the pipeline](#running-the-pipeline)).
- Participants without results are stored as NaN rather than zero, so that missing data cannot be mistaken for a genuine result.
- Random seeds are fixed for the random control and for Griffin-Lim so that results are reproducible.

## Example results

The figures from a full run are included in `results/figures/`.

- **Decoding.** The mean Pearson correlation between the original and decoded spectrograms, averaged over the 80 Mel bins and 10 folds, ranges from 0.46 (sub-01) to 0.82 (sub-06). Eight of the ten participants beat all 1,000 shifts (p ≈ 0.001), but sub-01 and sub-08 do not. These values are not directly comparable with Verwoert et al. (2022) because the target spectrogram has changed.
- **Electrode coverage.** Participants with more STG/vSMC contacts decoded better (r = 0.68, p = 0.029), but much of this depends on sub-06, which has 20 contacts against 0-5 for everyone else.
- **Vocoders on the ground-truth spectrogram.** HiFi-GAN scored higher on STOI than Griffin-Lim for all ten participants, whilst Griffin-Lim scored better (lower) on LSD and MCD for all ten (Holm-corrected p = 0.012 for each).
- **Vocoders on the decoded spectrogram.** The two vocoders did not differ significantly on any metric. For both, mean STOI falls from about 0.9 on the ground-truth spectrogram to about 0.35 on the decoded one, so most of the quality loss comes from decoding rather than vocoding.

![Acoustic benchmark of Griffin-Lim and HiFi-GAN](results/figures/acoustic_benchmarking_hifigan_vs_gl.png)

![Close-up spectrograms of one word for both vocoders](results/figures/vocoder_closeup.png)

## Next steps

- Replace the linear decoder with a non-linear model.
- Fine-tune HiFi-GAN on decoded spectrograms so that it learns to handle their smoother spectral shape.
- Work out which electrodes matter most using activation patterns rather than raw decoder weights, which can be misleading (Haufe et al., 2014).

## Repository structure

```
.
|-- step01_data_loader.py       # loads and checks the iBIDS dataset
|-- step02_extract_features.py  # high-gamma features and log-Mel spectrograms
|-- step03_train_decoder.py     # cross-validated decoding and random control
|-- step04_audio_synthesis.py   # Griffin-Lim and HiFi-GAN synthesis
|-- step05_results.py           # statistics, benchmarks and figures
|-- utilities/
|   |-- MelFilterBank.py        # Mel filterbank used by Griffin-Lim
|   `-- reconstructWave.py      # STFT, inverse STFT and Griffin-Lim
|-- results/
|   `-- figures/                # figures from a full run
|-- load_data.m                 # MATLAB loading example from the original authors
|-- requirements.txt
`-- LICENSE
```

## Getting the data

The dataset is not included in this repository. It is stored in the iEEG Brain Imaging Data Structure (iBIDS) format and can be downloaded from the Open Science Framework (OSF):

1. Go to [https://osf.io/nrgx6](https://osf.io/nrgx6/) and download the `SingleWordProductionDutch-iBIDS` folder.
2. Place the folder in the root of this repository, next to the step scripts.

The scripts expect the following structure:

```
SingleWordProductionDutch-iBIDS/
|-- participants.tsv
|-- sub-01/
|   `-- ieeg/
|       |-- sub-01_task-wordProduction_ieeg.nwb
|       |-- sub-01_task-wordProduction_channels.tsv
|       |-- sub-01_task-wordProduction_space-ACPC_electrodes.tsv
|       `-- ...
|-- sub-02/
|   `-- ...
`-- sub-10/
```

Please cite Verwoert et al. (2022) if you use the dataset.

## Installation

The pipeline was developed and tested on Windows with Python 3.14.6.

```bash
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

On Windows, `pip install` installs the CPU version of PyTorch. To run HiFi-GAN on a GPU, install the CUDA build of `torch` and `torchaudio` from [pytorch.org](https://pytorch.org/get-started/locally/) first, using the versions in `requirements.txt`.

The HiFi-GAN checkpoint ([speechbrain/tts-hifigan-libritts-16kHz](https://huggingface.co/speechbrain/tts-hifigan-libritts-16kHz)) is downloaded automatically into `pretrained_models/` the first time step 4 runs, so an internet connection is needed for that run.

## Running the pipeline

Run the scripts in order from the repository root. Each step reads the outputs of the steps before it.

```bash
python step01_data_loader.py
python step02_extract_features.py
python step03_train_decoder.py
python step04_audio_synthesis.py
python step05_results.py
```

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `step01_data_loader.py` | Loads the dataset, checks every participant (sampling rates, channel counts, invalid values, dead channels) and plots diagnostic figures for sub-01 | Console summary and diagnostic plots |
| 2 | `step02_extract_features.py` | Extracts high-gamma (70-170 Hz) features in 64 ms windows, adds ±192 ms of context and computes 80-bin log-Mel spectrograms | `features/` |
| 3 | `step03_train_decoder.py` | Decodes the spectrogram with PCA (50 components) and linear regression in 10-fold cross-validation and computes the random control | `results/` |
| 4 | `step04_audio_synthesis.py` | Synthesises audio from the ground-truth and predicted spectrograms with Griffin-Lim and HiFi-GAN | `results/audio/` |
| 5 | `step05_results.py` | Runs the statistical tests and acoustic benchmarks and creates all figures | `results/figures/` and console tables |

Step 1 only inspects the data, so it can be skipped once the dataset has been checked.

Steps 1 and 5 open each figure in a window. Close the window to let the script continue.

## Outputs

`step05_results.py` saves the following figures to `results/figures/`:

| File | Content |
|---|---|
| `results.png` | Mean reconstruction correlation per participant and per Mel bin, with chance levels from the random control |
| `channel_coverage_analysis.png` | Distribution of decoder weights across channels and decoding performance against STG/vSMC electrode coverage |
| `acoustic_benchmarking_hifigan_vs_gl.png` | STOI, LSD and MCD for Griffin-Lim and HiFi-GAN in the oracle and decoded conditions |
| `spec_example.png` | Original and reconstructed spectrograms for five words |
| `wav_example_comparison.png` | Original waveform and the decoded Griffin-Lim and HiFi-GAN waveforms for the same words |
| `vocoder_closeup.png` | Spectrograms of one word from the original recording and both vocoders in both conditions |

## Notes and limitations

- **Chance level.** Each trial shows a word for 2 s followed by a fixation cross for 1 s. A shift close to a multiple of 3 s lines speech up with speech again and gives a high correlation without any decoding, so the control is conservative.
- **Channel importance.** The decoder weights are mapped back from PCA space to the original features and each channel's importance is its mean absolute weight across Mel bins, context windows and folds. Weights of a linear decoder do not directly show which channels carry speech information (Haufe et al., 2014), so these values should be interpreted with caution.
- **MCD values.** MCD is computed from natural-log Mel spectra with an orthonormal DCT. Values are comparable within this project but not with MCD values in the speech synthesis literature, which are usually computed with the Speech Signal Processing Toolkit (SPTK).
- **Highest Mel bins.** The audio is downsampled from 48 kHz to 16 kHz with an anti-aliasing filter that removes most energy above about 6.4 kHz, so correlations drop in the highest Mel bins.
- **HiFi-GAN.** The vocoder is pretrained on English read speech (LibriTTS) and is not fine-tuned on Dutch or on decoded spectrograms.
- **Brain regions.** STG includes the lateral STG, planum temporale, planum polare and Heschl's gyrus. The Destrieux atlas does not separate ventral from dorsal sensorimotor cortex, so vSMC includes the subcentral gyrus and inferior precentral sulcus, plus precentral, postcentral and central sulcus contacts at or below an ACPC z-coordinate of 40 mm. This cut-off approximates the boundary between the face and larynx areas and the hand area.

## References

### Background

Papers on speech decoding and the cortical representation of speech that informed the design of this pipeline.

- Anumanchipalli, G.K., Chartier, J. and Chang, E.F. (2019) ‘Speech synthesis from neural decoding of spoken sentences’, Nature, 568(7753), pp. 493–498. Available at: https://doi.org/10.1038/s41586-019-1119-1.
- Bouchard, K.E. et al. (2013) ‘Functional organization of human sensorimotor cortex for speech articulation’, Nature, 495(7441), pp. 327–332. Available at: https://doi.org/10.1038/nature11911.
- Chartier, J. et al. (2018) ‘Encoding of Articulatory Kinematic Trajectories in Human Speech Sensorimotor Cortex’, Neuron, 98(5), pp. 1042-1054.e4. Available at: https://doi.org/10.1016/j.neuron.2018.04.031.
- He, T. et al. (2025) ‘VocalMind: A Stereotactic EEG Dataset for Vocalized, Mimed, and Imagined Speech in Tonal Language’, Scientific Data, 12(1), p. 657. Available at: https://doi.org/10.1038/s41597-025-04741-2.
- Liu, Y. et al. (2023) ‘Decoding and synthesizing tonal language speech from brain activity’, Science Advances, 9(23), p. eadh0478. Available at: https://doi.org/10.1126/sciadv.adh0478.
- Mesgarani, N. et al. (2014) ‘Phonetic Feature Encoding in Human Superior Temporal Gyrus’, Science, 343(6174), pp. 1006–1010. Available at: https://doi.org/10.1126/science.1245994.
- Metzger, S.L. et al. (2023) ‘A high-performance neuroprosthesis for speech decoding and avatar control’, Nature, 620(7976), pp. 1037–1046. Available at: https://doi.org/10.1038/s41586-023-06443-4.
- Silva, A.B. et al. (2024) ‘The speech neuroprosthesis’, Nature Reviews Neuroscience, 25(7), pp. 473–492. Available at: https://doi.org/10.1038/s41583-024-00819-9.
- Zhang, D. et al. (2024) ‘A brain-to-text framework for decoding natural tonal sentences’, Cell Reports, 43(11), p. 114924. Available at: https://doi.org/10.1016/j.celrep.2024.114924.
- Zhang, H. et al. (2026) ‘A Fine-grained Spatiotemporal ECoG Dataset during Speech Perception in Tonal Language’, Scientific Data, 13(1), p. 1380. Available at: https://doi.org/10.1038/s41597-026-07619-z.

### Methods and software

The dataset, methods and tools that the code directly uses.

- Destrieux, C. et al. (2010) ‘Automatic parcellation of human cortical gyri and sulci using standard anatomical nomenclature’, NeuroImage, 53(1), pp. 1–15. Available at: https://doi.org/10.1016/j.neuroimage.2010.06.010.
- Griffin, D. and Lim, J. (1984) ‘Signal estimation from modified short-time Fourier transform’, IEEE Transactions on Acoustics, Speech, and Signal Processing, 32(2), pp. 236–243. Available at: https://doi.org/10.1109/TASSP.1984.1164317.
- Haufe, S. et al. (2014) ‘On the interpretation of weight vectors of linear models in multivariate neuroimaging’, NeuroImage, 87, pp. 96–110. Available at: https://doi.org/10.1016/j.neuroimage.2013.10.067.
- Holm, S. (1979) ‘A Simple Sequentially Rejective Multiple Test Procedure’, Scandinavian Journal of Statistics, 6(2), pp. 65–70. Available at: https://doi.org/10.2307/4615733.
- Kong, J., Kim, J. and Bae, J. (2020) ‘HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis’. arXiv. Available at: https://doi.org/10.48550/arXiv.2010.05646.
- Phipson, B. and Smyth, G.K. (2010) ‘Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn’, Statistical Applications in Genetics and Molecular Biology, 9, Article 39. Available at: https://doi.org/10.2202/1544-6115.1585.
- Ravanelli, M. et al. (2021) ‘SpeechBrain: A General-Purpose Speech Toolkit’. arXiv. Available at: https://doi.org/10.48550/arXiv.2106.04624.
- Taal, C.H. et al. (2011) ‘An Algorithm for Intelligibility Prediction of Time–Frequency Weighted Noisy Speech’, IEEE Transactions on Audio, Speech, and Language Processing, 19(7), pp. 2125–2136. Available at: https://doi.org/10.1109/TASL.2011.2114881.
- Verwoert, M. et al. (2022) ‘Dataset of Speech Production in intracranial Electroencephalography’, Scientific Data, 9(1), p. 434. Available at: https://doi.org/10.1038/s41597-022-01542-9.
- Zen, H. et al. (2019) ‘LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech’. arXiv. Available at: https://doi.org/10.48550/arXiv.1904.02882.

## Licence

The original code was released by the Neural Interfacing Lab under the MIT License and this repository keeps the same licence (see `LICENSE`). Modifications by Haseeb Ahmad Irshad.

## Acknowledgements

I would like to thank the Advanced Biomedical Instrumentation Centre and my supervisors for their guidance and support during this project.