import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def calculate_pearson_r(x, y):
    """
    Compute the Pearson correlation coefficient between two 1D vectors with protection against zero variance

    Parameters
    ----------
    x : np.ndarray
        First 1D numeric feature or target data vector
    y : np.ndarray
        Second 1D numeric feature or target data vector

    Returns
    -------
    r : float
        Pearson correlation coefficient (returns 0.0 if either vector has zero or near-zero variance)
    """
    # guard against near-zero variance
    if np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return 0.0

    r, _ = pearsonr(x, y)

    return r

def reconstruct_spectrogram_cv(data, spectrogram, kf, numComps, modelOrder):
    """
    Reconstruct Mel spectrograms from iEEG features via cross-validated PCA and linear regression,
    computing channel importance weights back-projected to electrode space

    Parameters
    ----------
    data : np.ndarray
        Temporally stacked iEEG feature matrix formatted with shape (windows, features)
    spectrogram : np.ndarray
        Target Z-scored log-Mel spectrogram matrix formatted with shape (windows, mel_bins)
    kf : sklearn.model_selection.KFold
        Scikit-learn KFold cross-validation splitter instance
    numComps : int
        Maximum number of principal components retained during feature reduction
    modelOrder : int
        Temporal context stacking parameter defining past/future frame lags

    Returns
    -------
    rec_spec : np.ndarray
        Reconstructed target spectrogram compiled across test folds formatted with shape (windows, mel_bins)
    rs : np.ndarray
        Pearson correlation coefficients computed per fold and spectral bin formatted with shape (folds, mel_bins)
    exp_var : np.ndarray
        Cumulative explained variance ratio accounted for by PCA components per fold formatted with shape (folds,)
    channel_importance : np.ndarray
        Averaged spatial channel importance weights back-projected across features and temporal lags formatted with shape (channels,)
    """
    nfolds = kf.get_n_splits()
    num_bins = spectrogram.shape[1] # supports dynamic Mel bins

    # calculate the number of temporal context lags and physical electrode channels
    num_lags = 2 * modelOrder + 1
    num_features = data.shape[1]
    num_channels = num_features // num_lags

    # initialise matrices for reconstructed spectrogram, correlations, variance and channel weights
    rec_spec = np.zeros(spectrogram.shape)
    rs = np.zeros((nfolds, num_bins))
    exp_var = np.zeros(nfolds)
    fold_channel_importance = np.zeros((nfolds, num_channels))

    pca = PCA()
    est = LinearRegression(n_jobs = -1)

    for k, (train, test) in enumerate(kf.split(data)):
        # Z-normalise features using training set mean and std
        mu = np.mean(data[train, :], axis = 0)
        std = np.std(data[train, :], axis = 0)

        trainData = (data[train, :] - mu) / (std + 1e-8)
        testData = (data[test, :] - mu) / (std + 1e-8)

        # fit PCA on normalised training features
        pca.fit(trainData)

        # ensure PCA components do not exceed feature or sample dimensions
        actualComps = min(numComps, trainData.shape[0], trainData.shape[1])
        exp_var[k] = np.sum(pca.explained_variance_ratio_[:actualComps])

        # transform training and testing features to principal component space
        trainData_pca = np.dot(trainData, pca.components_[:actualComps, :].T)
        testData_pca = np.dot(testData, pca.components_[:actualComps, :].T)

        # fit ordinary least squares linear regression model and predict test fold
        est.fit(trainData_pca, spectrogram[train, :])
        rec_spec[test, :] = est.predict(testData_pca)

        # evaluate reconstruction accuracy per spectral bin
        for specBin in range(num_bins):
            rs[k, specBin] = calculate_pearson_r(spectrogram[test, specBin], rec_spec[test, specBin])

        # back-project linear model weights (num_bins x actualComps) through PCA components (actualComps x num_features)
        # est.coef_ shape: (num_bins, actualComps), pca.components_ shape: (actualComps, num_features)
        W_feat = np.dot(est.coef_, pca.components_[:actualComps, :]) # shape: (num_bins, num_features)

        # scale by feature standard deviation to return to raw feature amplitude units
        W_raw = W_feat / (std + 1e-8) # shape: (num_bins, num_features)

        # reshape weight matrix to separate physical channels from temporal context lags
        W_reshaped = W_raw.reshape(num_bins, num_lags, num_channels) # shape: (num_bins, num_lags, num_channels)

        # aggregate weight magnitude across spectral bins (axis 0) and temporal lags (axis 1)
        fold_channel_importance[k, :] = np.mean(np.abs(W_reshaped), axis = (0, 1))

    # average spatial channel importance weights across all cross-validation folds
    channel_importance = np.mean(fold_channel_importance, axis = 0)

    return rec_spec, rs, exp_var, channel_importance

def compute_random_control(spectrogram, numRands, rng = None):
    """
    Estimate empirical null baseline correlations using circular temporal shift permutations

    Parameters
    ----------
    spectrogram : np.ndarray
        Target Z-scored log-Mel spectrogram matrix formatted with shape (windows, mel_bins)
    numRands : int
        Number of random circular shift permutations to execute
    rng : np.random.Generator, optional
        NumPy random number generator instance for reproducibility (default is None, which creates
        an unseeded generator, so results will vary across runs)
    
    Returns
    -------
    randomControl : np.ndarray
        Matrix of empirical null baseline Pearson correlation values formatted with shape (numRands, mel_bins)
    """
    if rng is None:
        rng = np.random.default_rng()

    # cost scales as numRands x num_bins individual pearsonr calls (e.g. 1000 x 80 = 80,000 per participant) via this nested loop
    num_samples, num_bins = spectrogram.shape
    randomControl = np.zeros((numRands, num_bins))

    for randRound in range(numRands):
        # choose a random split point at least 10% away from boundaries
        splitPoint = rng.integers(int(num_samples * 0.1), int(num_samples * 0.9))

        # perform circular shift on target spectrogram
        shuffled = np.concatenate((spectrogram[splitPoint:, :], spectrogram[:splitPoint, :]))

        # calculate baseline correlations
        for specBin in range(num_bins):
            randomControl[randRound, specBin] = calculate_pearson_r(spectrogram[:, specBin], shuffled[:, specBin])

    return randomControl

if __name__ == "__main__":
    # ensure paths are built relative to this script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    feat_path = os.path.join(script_dir, 'features')
    result_path = os.path.join(script_dir, 'results')

    os.makedirs(result_path, exist_ok = True)

    pts = [f'sub-{i:02d}' for i in range(1, 11)]

    # cross-validation and regression parameters
    nfolds = 10
    kf = KFold(n_splits = nfolds, shuffle = False)
    numComps = 50
    numRands = 1000
    num_mel_bins = 80
    modelOrder = 4

    # fixed seed so the random control baseline is reproducible across runs
    rng = np.random.default_rng(30)

    # initialise NaN-filled matrices for correlation results, randomised controls and amount of explained variance
    allRes = np.full((len(pts), nfolds, num_mel_bins), np.nan)
    explainedVariance = np.full((len(pts), nfolds), np.nan)
    randomControl = np.full((len(pts), numRands, num_mel_bins), np.nan)

    for pNr, pt in enumerate(pts):
        # build the file paths for this participant
        spec_file = os.path.join(feat_path, f'{pt}_spec.npy')
        feat_file = os.path.join(feat_path, f'{pt}_feat.npy')

        # check that all required files exist before processing the participant
        if not os.path.exists(spec_file) or not os.path.exists(feat_file):
            print(f"Skipping {pt}: missing one or more required feature/spectrogram files.")
            continue

        # load stacked iEEG feature matrix and target spectrogram
        spectrogram = np.load(spec_file)
        data = np.load(feat_file)

        # perform cross-validated spectrogram reconstruction and channel importance estimation
        rec_spec, rs, exp_var, channel_importance = reconstruct_spectrogram_cv(data, spectrogram, kf, numComps, modelOrder)

        allRes[pNr, :, :] = rs
        explainedVariance[pNr, :] = exp_var

        # show evaluation result
        mean_r = np.mean(rs)
        print(f"Participant {pt}: Mean Reconstruction Pearson r = {mean_r:.4f}")

        # estimate empirical null random baseline controls
        rand_ctrl = compute_random_control(spectrogram, numRands, rng = rng)
        randomControl[pNr, :, :] = rand_ctrl

        # save per-participant spatial channel importance array and reconstructed spectrogram
        np.save(os.path.join(result_path, f'{pt}_channel_importance.npy'), channel_importance)
        np.save(os.path.join(result_path, f'{pt}_predicted_spec.npy'), rec_spec)

    # save cross-participant aggregated benchmark matrices
    np.save(os.path.join(result_path, 'linearResults.npy'), allRes)
    np.save(os.path.join(result_path, 'randomResults.npy'), randomControl)
    np.save(os.path.join(result_path, 'explainedVariance.npy'), explainedVariance)