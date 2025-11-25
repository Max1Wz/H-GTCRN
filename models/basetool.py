# Author: Xianrui Wang
# Contact: wangxianrui@mail.nwpu.edu.cn
# Notice even the axes order is inflexible for variants of IVA,
# it suits fastMNMF

from tkinter import Y
import librosa
import numpy as np
import soundfile as sf
#from mir_eval.separation import bss_eval_sources, bss_eval_images


def multichannel_stft(mixed_signal=None, nfft=None, hop=None):
    """

    Parameters
    ----------
    data: input data M*T, notice T is time length
    n_fft: block of fft
    hop_length: hop size
    win_length: length of window

    Returns
    -------
    spec_FTM: multichannel spectrogram F*T*M, notice T means stft frames
    """
    M = mixed_signal.shape[0]
    for m in range(M):
        tmp = librosa.core.stft(mixed_signal[m], win_length=nfft, n_fft=nfft, hop_length=hop)
        if m == 0:
            spec_FTM = np.zeros([tmp.shape[0], tmp.shape[1], M], dtype=np.complex64)
        spec_FTM[:, :, m] = tmp
    return spec_FTM


def multichannel_istft(spec_FTM, nfft=None, hop=None):
    """
    Parameters
    ----------
    spectrogram: input multichannel spectrogram T*F*M, notice T means stft frames
    hop_length: hop size
    win_length: length of window
    ori_length: length of original length

    Returns
    -------
    y: multichannel time domain signal M*T , notice T is time length
    """
    N = spec_FTM.shape[2]
    for n in range(N):
        tmp = librosa.istft(spec_FTM[..., n], win_length=nfft, hop_length=hop)

        if n == 0:
            y = np.zeros([N, tmp.shape[0]])
        y[n] = tmp
    return y


def add_noise(multichannel_signal, SNR):
    """
    Add multichannel noise with given signal-to-noise ratio (SNR)
    """
    channel, siglength = multichannel_signal.shape
    random_values = np.random.rand(channel, siglength)
    Ps = np.mean(multichannel_signal ** 2, axis=1)
    Pn1 = np.mean(random_values ** 2, axis=1)
    print(Ps.shape, Pn1.shape)
    k = (np.sqrt(Ps/(10**(SNR/10)*Pn1)))[:, None]
    random_values_we_need = random_values*k
    outdata = multichannel_signal + random_values_we_need
    return outdata


def performance_measure(s_image, y, sHat):
    sdr_mix, sir_mix, sar_mix, perm_mix = bss_eval_sources(s_image, y, compute_permutation=True)
    sdr, sir, sar, perm = bss_eval_sources(s_image, sHat, compute_permutation=True)
    sdr_improve = sdr-sdr_mix
    sir_improve = sir-sir_mix
    return sdr_mix, sir, sdr_mix, sdr, sdr_improve, sir_improve


def projection_back(Y, ref, clip_up=None, clip_down=None):
    """
    This function computes the frequency-domain filter that minimizes
    the squared error to a reference signal. This is commonly used
    to solve the scale ambiguity in BSS.

    Here is the derivation of the projection.
    The optimal filter `z` minimizes the squared error.

    .. math::

        \min E[|z^* y - x|^2]

    It should thus satsify the orthogonality condition
    and can be derived as follows

    .. math::

        0 & = E[y^*\\, (z^* y - x)]

        0 & = z^*\\, E[|y|^2] - E[y^* x]

        z^* & = \\frac{E[y^* x]}{E[|y|^2]}

        z & = \\frac{E[y x^*]}{E[|y|^2]}

    In practice, the expectations are replaced by the sample
    mean.

    Parameters
    ----------
    Y: array_like (n_frames, n_bins, n_channels)
        The STFT data to project back on the reference signal
    ref: array_like (n_frames, n_bins)
        The reference signal
    clip_up: float, optional
        Limits the maximum value of the gain (default no limit)
    clip_down: float, optional
        Limits the minimum value of the gain (default no limit)
    """

    num = np.sum(np.conj(ref[:, :, None]) * Y, axis=0)
    denom = np.sum(np.abs(Y) ** 2, axis=0)

    c = np.ones(num.shape, dtype=np.complex64)
    I = denom > 0.0
    c[I] = num[I] / denom[I]

    if clip_up is not None:
        I = np.logical_and(np.abs(c) > clip_up, np.abs(c) > 0)
        c[I] *= clip_up / np.abs(c[I])

    if clip_down is not None:
        I = np.logical_and(np.abs(c) < clip_down, np.abs(c) > 0)
        c[I] *= clip_down / np.abs(c[I])
    return c




def pca(X_FTM, n_src=None, return_filters=False, normalize=True):
    """
    Whitens the input signal X using principal component analysis (PCA)
    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        The input signal
    n_src: int
        The desired number of principal components
    return_filters: bool
        If this flag is set to true, the PCA matrix
        is also returned (default False)
    normalize: bool
        If this flag is set to false, the decorrelated
        channels are not normalized to unit variance
        (default True)
    """
    X_FMT = X_FTM.transpose([0, 2, 1])
    _, n_chan, n_frames = X_FMT.shape

    # default to determined case
    if n_src is None:
        n_src = n_chan

    assert (
        n_src <= n_chan
    ), "The number of sources cannot be more than the number of channels."

    # compute the cov mat (n_freq, n_chan, n_chan)
    covmat = (X_FMT @ np.conj(X_FTM)) * (1.0 / n_frames)
    # make sure the covmat is hermitian symmetric and positive semi-definite
    covmat = 0.5 * (covmat + np.conj(covmat.swapaxes(1,2)))
    # Compute EVD
    # v.shape == (n_freq, n_chan), w.shape == (n_freq, n_chan, n_chan)
    eig_val, eig_vec = np.linalg.eigh(covmat)

    # Reorder the eigenvalues from so that they are in descending order
    eig_val = eig_val[:, ::-1]
    eig_vec = eig_vec[:, :, ::-1]

    # The whitening matrices
    if normalize:
        Q = (1.0 / np.sqrt(eig_val[:, :, None])) * np.conj(eig_vec.swapaxes(1,2))
    else:
        Q = eig_vec.swapaxes(1,2)

    # The decorrelated signal
    Y_FNT = (Q[:, :n_src, :] @ X_FMT)
    Y_FTN = Y_FNT.swapaxes(1,2)


    if return_filters:
        return Y_FTN, Q
    else:
        return Y_FTN

