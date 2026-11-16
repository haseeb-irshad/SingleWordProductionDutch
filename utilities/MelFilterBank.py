import math
import numpy as np

class MelFilterBank():
    """
    Constructs and applies forward and inverse Mel filterbank transformation matrices
    for perceptual audio feature extraction and spectral resynthesis
    """

    def __init__(self, specSize, numCoefficients, sampleRate):
        """
        Initialise the MelFilterBank instance and pre-compute normalised forward
        and inverse transformation matrices

        Parameters
        ----------
        specSize : int
            Number of positive frequency bins in the STFT spectrum (N_fft / 2 + 1)
        numCoefficients : int or float
            Target number of Mel frequency bands (e.g. 80)
        sampleRate : int or float
            Sampling rate of the audio signal in Hz (e.g. 16000)
        """
        numBands = int(numCoefficients)

        # compute Mel-scale boundary points across the frequency spectrum
        minMel = 0 # start at 0 Mel (0 Hz)
        maxMel = self.freqToMel(sampleRate / 2.0) # top edge is the Nyquist frequency
        melStep = (maxMel - minMel) / (numBands + 1) # equal spacing step on the Mel scale

        # array of (numBands + 2) edge points defining start, centre and end for all filters
        melFilterEdges = np.arange(0, numBands + 2) * melStep

        # convert the Mel edge points to Hz and then to STFT bin indices
        centerIndices = list(map(
            lambda x: self.freqToBin(self.melToFreq(x), sampleRate, specSize), melFilterEdges
        ))

        # initialise an empty matrix of shape (numBands, specSize) to store filter weights
        filterMatrix = np.zeros((numBands, specSize), dtype = np.float64)

        # construct triangular filter weights for each Mel band
        for i in range(numBands):
            # extract start, peak (centre) and end bin indices for the i-th triangular filter
            start, center, end = centerIndices[i : i + 3]

            # width of the rising slope (k1) and falling slope (k2) in bin units
            k1 = np.float64(center - start)
            k2 = np.float64(end - center)

            # prevent division-by-zero warnings if adjacent bin indices coincide
            if k1 == 0:
                k1 = 1.0
            if k2 == 0:
                k2 = 1.0

            # compute linear slopes from 0.0 to 1.0 (rising) and 1.0 to 0.0 (falling)
            up = (np.arange(start, center) - start) / k1
            down = (end - np.arange(center, end)) / k2

            # assign slope weights into the matrix row
            filterMatrix[i, start:center] = up
            filterMatrix[i, center:end] = down

        # transpose and normalise forward matrix (linear STFT bins to Mel bands)
        # transpose shape to (specSize, numBands) for direct matrix multiplication
        self.melMatrix = filterMatrix.transpose()

        # normalise each column so that filter weights sum to 1.0 (weighted averaging)
        self.melMatrix = self.makeNormal(self.melMatrix / self.normSum(self.melMatrix))

        # construct inverse matrix (Mel bands to linear STFT bins)
        # transpose back to (numBands, specSize)
        self.melInvMatrix = self.melMatrix.transpose()

        # normalise each column so that the weights for every STFT bin sum to 1.0 (partition of unity)
        self.melInvMatrix = self.makeNormal(self.melInvMatrix / self.normSum(self.melInvMatrix))
    
    def normSum(self, x):
        """
        Calculate column-wise sums across a matrix, replacing zero sums with 1.0
        to prevent division-by-zero errors during normalisation
        """
        retSum = np.sum(x, axis = 0)
        retSum[retSum == 0] = 1.0
        return retSum
    
    def fuzz(self, x):
        """
        Add a small positive constant (1e-7) to numerical values to prevent log(0)
        """
        return x + 1e-7
    
    def freqToBin(self, freq, sampleRate, specSize):
        """
        Map a continuous frequency value in Hz to its discrete STFT bin index
        """
        return int(math.floor((freq / (sampleRate / 2.0)) * (specSize - 1)))
    
    def freqToMel(self, freq):
        """
        Convert frequency in Hz to perceptual pitch in Mels using the Slaney formula
        """
        min_log_hz = 1000.0
        min_log_mel = 15.0
        logstep = 27.0 / math.log(6.4)

        if freq >= min_log_hz:
            return min_log_mel + math.log(freq / min_log_hz) * logstep
        else:
            return 3.0 * freq / 200.0

    def melToFreq(self, mel):
        """
        Convert perceptual pitch in Mels back to frequency in Hz using the Slaney formula
        """
        min_log_hz = 1000.0
        min_log_mel = 15.0
        logstep = 27.0 / math.log(6.4)

        if mel >= min_log_mel:
            return min_log_hz * math.exp((mel - min_log_mel) / logstep)
        else:
            return 200.0 * mel / 3.0

    def makeNormal(self, x):
        """
        Sanitise numerical matrices by replacing NaN and Inf values with 0.0
        """
        x = np.nan_to_num(x, nan = 0.0, posinf = 0.0, neginf = 0.0)
        return x

    def toMelScale(self, spectrogram):
        """
        Convert a linear STFT magnitude spectrogram to the linear Mel scale

        Input matrix shape is (time frames, frequency bins)
        Output matrix shape is (time frames, Mel bands)
        """
        return np.dot(spectrogram, self.melMatrix)
    
    def fromMelScale(self, melSpectrogram):
        """
        Convert a linear Mel spectrogram back to a linear STFT magnitude spectrogram

        Input matrix shape is (time frames, Mel bands)
        Output matrix shape is (time frames, frequency bins)
        """
        return np.dot(melSpectrogram, self.melInvMatrix)
    
    def toMels(self, spectrogram):
        """
        Alias for toMelScale
        """
        return self.toMelScale(spectrogram)
    
    def fromMels(self, melSpectrogram):
        """
        Alias for fromMelScale
        """
        return self.fromMelScale(melSpectrogram)
    
    def toLogMels(self, spectrogram):
        """
        Convert a linear STFT spectrogram to a log-compressed Mel representation
        """
        return self.makeNormal(np.log(self.fuzz(self.toMelScale(spectrogram))))
    
    def fromLogMels(self, melSpectrogram):
        """
        Convert a log-compressed Mel spectrogram back to a linear STFT spectrogram
        """
        return self.makeNormal(self.fromMelScale(np.maximum(np.exp(melSpectrogram) - 1e-7, 0.0)))