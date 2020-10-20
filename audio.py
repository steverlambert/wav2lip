import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
import scipy
from scipy import signal
from scipy.io import wavfile
from hparams import hparams as hp
import soundfile as sf
from numpy.lib.stride_tricks import as_strided


MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10

def load_wav(path, sr, rosa=False):
    if rosa:
      return librosa.core.load(path, sr=sr, res_type='scipy')[0]
    else:
      return load(path, sr=sr, res_type='scipy')[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size

def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def _lws_processor():
    import lws
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")

def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None

def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)

def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)

def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
    
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))

def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
    
    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)



def load(
    path,
    sr=22050,
    mono=True,
    offset=0.0,
    duration=None,
    dtype=np.float32,
    res_type="scipy",):

    try:
        with sf.SoundFile(path) as sf_desc:
            sr_native = sf_desc.samplerate
            if offset:
                # Seek to the start of the target read
                sf_desc.seek(int(offset * sr_native))
            if duration is not None:
                frame_duration = int(duration * sr_native)
            else:
                frame_duration = -1

            # Load the target number of frames, and transpose to match librosa form
            y = sf_desc.read(frames=frame_duration, dtype=dtype, always_2d=False).T

    except RuntimeError as exc:
        # If soundfile failed, try audioread instead
        if isinstance(path, (str, pathlib.PurePath)):
            warnings.warn("PySoundFile failed. Trying audioread instead.")
            y, sr_native = __audioread_load(path, offset, duration, dtype)
        else:
            raise (exc)

    # Final cleanup for dtype and contiguity
    if mono:
        y = to_mono(y)

    if sr is not None:
        y = resample(y, sr_native, sr, res_type=res_type)
    else:
        sr = sr_native

    return y, sr

def to_mono(y):
    """Convert an audio signal to mono by averaging samples across channels.
    Parameters
    ----------
    y : np.ndarray [shape=(2,n) or shape=(n,)]
        audio time series, either stereo or mono
    Returns
    -------
    y_mono : np.ndarray [shape=(n,)]
        ``y`` as a monophonic time-series
    Notes
    -----
    This function caches at level 20.
    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
    >>> y.shape
    (2, 117601)
    >>> y_mono = librosa.to_mono(y)
    >>> y_mono.shape
    (117601,)
    """
    # Ensure Fortran contiguity.
    y = np.asfortranarray(y)

    # Validate the buffer.  Stereo is ok here.
    valid_audio(y, mono=False)

    if y.ndim > 1:
        y = np.mean(y, axis=0)

    return y

def valid_audio(y, mono=True):
    """Determine whether a variable contains valid audio data.
    If ``mono=True``, then ``y`` is only considered valid if it has shape
    ``(N,)`` (number of samples).
    If ``mono=False``, then ``y`` may be either monophonic, or have shape
    ``(2, N)`` (stereo) or ``(K, N)`` for ``K>=2`` for general multi-channel.
    Parameters
    ----------
    y : np.ndarray
      The input data to validate
    mono : bool
      Whether or not to require monophonic audio
    Returns
    -------
    valid : bool
        True if all tests pass
    Raises
    ------
    ParameterError
        In any of these cases:
            - ``type(y)`` is not ``np.ndarray``
            - ``y.dtype`` is not floating-point
            - ``mono == True`` and ``y.ndim`` is not 1
            - ``mono == False`` and ``y.ndim`` is not 1 or 2
            - ``mono == False`` and ``y.ndim == 2`` but ``y.shape[0] == 1``
            - ``np.isfinite(y).all()`` is False
    Notes
    -----
    This function caches at level 20.
    Examples
    --------
    >>> # By default, valid_audio allows only mono signals
    >>> filepath = librosa.ex('trumpet', hq=True)
    >>> y_mono, sr = librosa.load(filepath, mono=True)
    >>> y_stereo, _ = librosa.load(filepath, mono=False)
    >>> librosa.util.valid_audio(y_mono), librosa.util.valid_audio(y_stereo)
    True, False
    >>> # To allow stereo signals, set mono=False
    >>> librosa.util.valid_audio(y_stereo, mono=False)
    True
    See also
    --------
    numpy.float32
    """


    if not isinstance(y, np.ndarray):
        raise ParameterError("Audio data must be of type numpy.ndarray")

    if not np.issubdtype(y.dtype, np.floating):
        raise ParameterError("Audio data must be floating-point")

    if mono and y.ndim != 1:
        raise ParameterError(
            "Invalid shape for monophonic audio: "
            "ndim={:d}, shape={}".format(y.ndim, y.shape)
        )

    elif y.ndim > 2 or y.ndim == 0:
        raise ParameterError(
            "Audio data must have shape (samples,) or (channels, samples). "
            "Received shape={}".format(y.shape)
        )

    elif y.ndim == 2 and y.shape[0] < 2:
        raise ParameterError(
            "Mono data must have shape (samples,). " "Received shape={}".format(y.shape)
        )

    if not np.isfinite(y).all():
        raise ParameterError("Audio buffer is not finite everywhere")

    return True


def resample(
    y, orig_sr, target_sr, res_type="kaiser_best", fix=False, scale=False, **kwargs
):
    """Resample a time series from orig_sr to target_sr
    By default, this uses a high-quality (but relatively slow) method ('kaiser_best')
    for band-limited sinc interpolation.  The alternate ``res_type`` values listed below
    offer different trade-offs of speed and quality.
    Parameters
    ----------
    y : np.ndarray [shape=(n,) or shape=(2, n)]
        audio time series.  Can be mono or stereo.
    orig_sr : number > 0 [scalar]
        original sampling rate of ``y``
    target_sr : number > 0 [scalar]
        target sampling rate
    res_type : str
        resample type (see note)
        .. note::
            By default, this uses `resampy`'s high-quality mode ('kaiser_best').
            To use a faster method, set ``res_type='kaiser_fast'``.
            To use `scipy.signal.resample`, set ``res_type='fft'`` or ``res_type='scipy'``. (slow)
            To use `scipy.signal.resample_poly`, set ``res_type='polyphase'``. (fast)
            To use `samplerate.converters.resample`, set any of the following:
                - ``res_type='linear'``: linear interpolation (fast)
                - ``res_type='zero_order_hold'``: repeat the last value between samples (very fast)
                - ``res_type='sinc_best'``, ``'sinc_medium'``, or ``'sinc_fastest'``: for high-, medium-,
                  and low-quality sinc interpolation
        .. note::
            When using ``res_type='polyphase'``, only integer sampling rates are
            supported.
    fix : bool
        adjust the length of the resampled signal to be of size exactly
        ``ceil(target_sr * len(y) / orig_sr)``
    scale : bool
        Scale the resampled signal so that ``y`` and ``y_hat`` have approximately
        equal total energy.
    kwargs : additional keyword arguments
        If ``fix==True``, additional keyword arguments to pass to
        `librosa.util.fix_length`.
    Returns
    -------
    y_hat : np.ndarray [shape=(n * target_sr / orig_sr,)]
        ``y`` resampled from ``orig_sr`` to ``target_sr``
    Raises
    ------
    ParameterError
        If ``res_type='polyphase'`` and ``orig_sr`` or ``target_sr`` are not both
        integer-valued.
    See Also
    --------
    librosa.util.fix_length
    scipy.signal.resample
    resampy
    samplerate.converters.resample
    Notes
    -----
    This function caches at level 20.
    Examples
    --------
    Downsample from 22 KHz to 8 KHz
    >>> y, sr = librosa.load(librosa.ex('trumpet'), sr=22050)
    >>> y_8k = librosa.resample(y, sr, 8000)
    >>> y.shape, y_8k.shape
    ((117601,), (42668,))
    """

    # First, validate the audio buffer
    valid_audio(y, mono=False)

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[-1] * ratio))

    if res_type in ("scipy", "fft"):
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)
        print("yhat: ", y_hat)
    elif res_type == "polyphase":
        if int(orig_sr) != orig_sr or int(target_sr) != target_sr:
            raise ParameterError(
                "polyphase resampling is only supported for integer-valued sampling rates."
            )

        # For polyphase resampling, we need up- and down-sampling ratios
        # We can get those from the greatest common divisor of the rates
        # as long as the rates are integrable
        orig_sr = int(orig_sr)
        target_sr = int(target_sr)
        gcd = np.gcd(orig_sr, target_sr)
        y_hat = scipy.signal.resample_poly(y, target_sr // gcd, orig_sr // gcd, axis=-1)
    elif res_type in (
        "linear",
        "zero_order_hold",
        "sinc_best",
        "sinc_fastest",
        "sinc_medium",
    ):
        import samplerate

        # We have to transpose here to match libsamplerate
        y_hat = samplerate.resample(y.T, ratio, converter_type=res_type).T
    else:
        y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.asfortranarray(y_hat, dtype=y.dtype)


def stft(y,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=None,
    pad_mode="reflect",
):
 # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        if n_fft > y.shape[-1]:
            warnings.warn(
                "n_fft={} is too small for input signal of length={}".format(
                    n_fft, y.shape[-1]
                )
            )

        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    elif n_fft > y.shape[-1]:
        raise ParameterError(
            "n_fft={} is too small for input signal of length={}".format(
                n_fft, y.shape[-1]
            )
        )

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    if dtype is None:
        dtype = dtype_r2c(y.dtype)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty(
        (int(1 + n_fft // 2), y_frames.shape[1]), dtype=dtype, order="F"
    )

    fft = get_fftlib()

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft.rfft(
            fft_window * y_frames[:, bl_s:bl_t], axis=0
        )
    return stft_matrix

def pad_center(data, size, axis=-1, **kwargs):
    """Pad an array to a target length along a target axis.
    This differs from `np.pad` by centering the data prior to padding,
    analogous to `str.center`
    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])
    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])
    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered
    size : int >= len(data) [scalar]
        Length to pad ``data``
    axis : int
        Axis along which to pad and center the data
    kwargs : additional keyword arguments
      arguments passed to `np.pad`
    Returns
    -------
    data_padded : np.ndarray
        ``data`` centered and padded to length ``size`` along the
        specified axis
    Raises
    ------
    ParameterError
        If ``size < data.shape[axis]``
    See Also
    --------
    numpy.pad
    """

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(
            ("Target size ({:d}) must be " "at least input size ({:d})").format(size, n)
        )

    return np.pad(data, lengths, **kwargs)

def frame(x, frame_length, hop_length, axis=-1):
    """Slice a data array into (overlapping) frames.
    This implementation uses low-level stride manipulation to avoid
    making a copy of the data.  The resulting frame representation
    is a new view of the same input data.
    However, if the input data is not contiguous in memory, a warning
    will be issued and the output will be a full copy, rather than
    a view of the input data.
    For example, a one-dimensional input ``x = [0, 1, 2, 3, 4, 5, 6]``
    can be framed with frame length 3 and hop length 2 in two ways.
    The first (``axis=-1``), results in the array ``x_frames``::
        [[0, 2, 4],
         [1, 3, 5],
         [2, 4, 6]]
    where each column ``x_frames[:, i]`` contains a contiguous slice of
    the input ``x[i * hop_length : i * hop_length + frame_length]``.
    The second way (``axis=0``) results in the array ``x_frames``::
        [[0, 1, 2],
         [2, 3, 4],
         [4, 5, 6]]
    where each row ``x_frames[i]`` contains a contiguous slice of the input.
    This generalizes to higher dimensional inputs, as shown in the examples below.
    In general, the framing operation increments by 1 the number of dimensions,
    adding a new "frame axis" either to the end of the array (``axis=-1``)
    or the beginning of the array (``axis=0``).
    Parameters
    ----------
    x : np.ndarray
        Array to frame
    frame_length : int > 0 [scalar]
        Length of the frame
    hop_length : int > 0 [scalar]
        Number of steps to advance between frames
    axis : 0 or -1
        The axis along which to frame.
        If ``axis=-1`` (the default), then ``x`` is framed along its last dimension.
        ``x`` must be "F-contiguous" in this case.
        If ``axis=0``, then ``x`` is framed along its first dimension.
        ``x`` must be "C-contiguous" in this case.
    Returns
    -------
    x_frames : np.ndarray [shape=(..., frame_length, N_FRAMES) or (N_FRAMES, frame_length, ...)]
        A framed view of ``x``, for example with ``axis=-1`` (framing on the last dimension)::
            x_frames[..., j] == x[..., j * hop_length : j * hop_length + frame_length]
        If ``axis=0`` (framing on the first dimension), then::
            x_frames[j] = x[j * hop_length : j * hop_length + frame_length]
    Raises
    ------
    ParameterError
        If ``x`` is not an `np.ndarray`.
        If ``x.shape[axis] < frame_length``, there is not enough data to fill one frame.
        If ``hop_length < 1``, frames cannot advance.
        If ``axis`` is not 0 or -1.  Framing is only supported along the first or last axis.
    See Also
    --------
    numpy.asfortranarray : Convert data to F-contiguous representation
    numpy.ascontiguousarray : Convert data to C-contiguous representation
    numpy.ndarray.flags : information about the memory layout of a numpy `ndarray`.
    Examples
    --------
    Extract 2048-sample frames from monophonic signal with a hop of 64 samples per frame
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    >>> frames
    array([[-1.407e-03, -2.604e-02, ..., -1.795e-05, -8.108e-06],
           [-4.461e-04, -3.721e-02, ..., -1.573e-05, -1.652e-05],
           ...,
           [ 7.960e-02, -2.335e-01, ..., -6.815e-06,  1.266e-05],
           [ 9.568e-02, -1.252e-01, ...,  7.397e-06, -1.921e-05]],
          dtype=float32)
    >>> y.shape
    (117601,)
    >>> frames.shape
    (2048, 1806)
    Or frame along the first axis instead of the last:
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64, axis=0)
    >>> frames.shape
    (1806, 2048)
    Frame a stereo signal:
    >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
    >>> y.shape
    (2, 117601)
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    (2, 2048, 1806)
    Carve an STFT into fixed-length patches of 32 frames with 50% overlap
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> S.shape
    (1025, 230)
    >>> S_patch = librosa.util.frame(S, frame_length=32, hop_length=16)
    >>> S_patch.shape
    (1025, 32, 13)
    >>> # The first patch contains the first 32 frames of S
    >>> np.allclose(S_patch[:, :, 0], S[:, :32])
    True
    >>> # The second patch contains frames 16 to 16+32=48, and so on
    >>> np.allclose(S_patch[:, :, 1], S[:, 16:48])
    True
    """

    if not isinstance(x, np.ndarray):
        raise ParameterError(
            "Input must be of type numpy.ndarray, " "given type(x)={}".format(type(x))
        )

    if x.shape[axis] < frame_length:
        raise ParameterError(
            "Input is too short (n={:d})"
            " for frame_length={:d}".format(x.shape[axis], frame_length)
        )

    if hop_length < 1:
        raise ParameterError("Invalid hop_length: {:d}".format(hop_length))

    if axis == -1 and not x.flags["F_CONTIGUOUS"]:
        warnings.warn(
            "librosa.util.frame called with axis={} "
            "on a non-contiguous input. This will result in a copy.".format(axis)
        )
        x = np.asfortranarray(x)
    elif axis == 0 and not x.flags["C_CONTIGUOUS"]:
        warnings.warn(
            "librosa.util.frame called with axis={} "
            "on a non-contiguous input. This will result in a copy.".format(axis)
        )
        x = np.ascontiguousarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize

    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]

    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)

    else:
        raise ParameterError("Frame axis={} must be either 0 or -1".format(axis))

    return as_strided(x, shape=shape, strides=strides)


def get_window(window, Nx, fftbins=True):
    """Compute a window function.
    This is a wrapper for `scipy.signal.get_window` that additionally
    supports callable or pre-computed windows.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        The window specification:
        - If string, it's the name of the window function (e.g., `'hann'`)
        - If tuple, it's the name of the window function and any parameters
          (e.g., `('kaiser', 4.0)`)
        - If numeric, it is treated as the beta parameter of the `'kaiser'`
          window, as in `scipy.signal.get_window`.
        - If callable, it's a function that accepts one integer argument
          (the window length)
        - If list-like, it's a pre-computed window of the correct length `Nx`
    Nx : int > 0
        The length of the window
    fftbins : bool, optional
        If True (default), create a periodic window for use with FFT
        If False, create a symmetric window for filter design applications.
    Returns
    -------
    get_window : np.ndarray
        A window of length `Nx` and type `window`
    See Also
    --------
    scipy.signal.get_window
    Notes
    -----
    This function caches at level 10.
    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length != `n_fft`,
        or is otherwise mis-specified.
    """
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ParameterError(
            "Window size mismatch: " "{:d} != {:d}".format(len(window), Nx)
        )
    else:
        raise ParameterError("Invalid window specification: {}".format(window))


def fix_length(data, size, axis=-1, **kwargs):
    """Fix the length an array ``data`` to exactly ``size`` along a target axis.
    If ``data.shape[axis] < n``, pad according to the provided kwargs.
    By default, ``data`` is padded with trailing zeros.
    Examples
    --------
    >>> y = np.arange(7)
    >>> # Default: pad with zeros
    >>> librosa.util.fix_length(y, 10)
    array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0])
    >>> # Trim to a desired length
    >>> librosa.util.fix_length(y, 5)
    array([0, 1, 2, 3, 4])
    >>> # Use edge-padding instead of zeros
    >>> librosa.util.fix_length(y, 10, mode='edge')
    array([0, 1, 2, 3, 4, 5, 6, 6, 6, 6])
    Parameters
    ----------
    data : np.ndarray
      array to be length-adjusted
    size : int >= 0 [scalar]
      desired length of the array
    axis : int, <= data.ndim
      axis along which to fix length
    kwargs : additional keyword arguments
        Parameters to ``np.pad``
    Returns
    -------
    data_fixed : np.ndarray [shape=data.shape]
        ``data`` either trimmed or padded to length ``size``
        along the specified axis.
    See Also
    --------
    numpy.pad
    """

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

def dtype_r2c(d, default=np.complex64):
    """Find the complex numpy dtype corresponding to a real dtype.
    This is used to maintain numerical precision and memory footprint
    when constructing complex arrays from real-valued data
    (e.g. in a Fourier transform).
    A `float32` (single-precision) type maps to `complex64`,
    while a `float64` (double-precision) maps to `complex128`.
    Parameters
    ----------
    d : np.dtype
        The real-valued dtype to convert to complex.
        If ``d`` is a complex type already, it will be returned.
    default : np.dtype, optional
        The default complex target type, if ``d`` does not match a
        known dtype
    Returns
    -------
    d_c : np.dtype
        The complex dtype
    See Also
    --------
    dtype_c2r
    numpy.dtype
    Examples
    --------
    >>> librosa.util.dtype_r2c(np.float32)
    dtype('complex64')
    >>> librosa.util.dtype_r2c(np.int16)
    dtype('complex64')
    >>> librosa.util.dtype_r2c(np.complex128)
    dtype('complex128')
    """
    mapping = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
        np.dtype(np.float): np.complex,
    }

    # If we're given a complex type already, return it
    dt = np.dtype(d)
    if dt.kind == "c":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))

"""Fast Fourier Transform (FFT) library container"""


__all__ = ["get_fftlib", "set_fftlib"]

# Object to hold FFT interfaces
__FFTLIB = None


def set_fftlib(lib=None):
    """Set the FFT library used by librosa.
    Parameters
    ----------
    lib : None or module
        Must implement an interface compatible with `numpy.fft`.
        If ``None``, reverts to `numpy.fft`.
    Examples
    --------
    Use `pyfftw`:
    >>> import pyfftw
    >>> librosa.set_fftlib(pyfftw.interfaces.numpy_fft)
    Reset to default `numpy` implementation
    >>> librosa.set_fftlib()
    """

    global __FFTLIB
    if lib is None:
        from numpy import fft

        lib = fft

    __FFTLIB = lib


def get_fftlib():
    """Get the FFT library currently used by librosa
    Returns
    -------
    fft : module
        The FFT library currently used by librosa.
        Must API-compatible with `numpy.fft`.
    """
    global __FFTLIB
    return __FFTLIB


# Set the FFT library to numpy's, by default
set_fftlib(None)



def mel(
    sr,
    n_fft,
    n_mels=128,
    fmin=0.0,
    fmax=None,
    htk=False,
    norm="slaney",
    dtype=np.float32,
):
    """Create a Mel filter-bank.
    This produces a linear transformation matrix to project 
    FFT bins onto Mel-frequency bins.
    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal
    n_fft     : int > 0 [scalar]
        number of FFT components
    n_mels    : int > 0 [scalar]
        number of Mel bands to generate
    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    htk       : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 'slaney', or number} [scalar]
        If 'slaney', divide the triangular mel weights by the width of the mel band
        (area normalization).
        If numeric, use `librosa.util.normalize` to normalize each filter by to unit l_p norm.
        See `librosa.util.normalize` for a full description of supported norm values
        (including `+-np.inf`).
        Otherwise, leave all the triangles aiming for a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.
    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix
    See also
    --------
    librosa.util.normalize
    Notes
    -----
    This function caches at level 10.
    Examples
    --------
    >>> melfb = librosa.filters.mel(22050, 2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])
    Clip the maximum frequency to 8KHz
    >>> librosa.filters.mel(22050, 2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(melfb, x_axis='linear', ax=ax)
    >>> ax.set(ylabel='Mel filter', title='Mel filter bank')
    >>> fig.colorbar(img, ax=ax)
    """

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    else:
        weights = normalize(weights, norm=norm, axis=-1)

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn(
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels."
        )

    return weights


def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
    """Normalize an array along a chosen axis.
    Given a norm (described below) and a target axis, the input
    array is scaled so that::
        norm(S, axis=axis) == 1
    For example, ``axis=0`` normalizes each column of a 2-d array
    by aggregating over the rows (0-axis).
    Similarly, ``axis=1`` normalizes each row of a 2-d array.
    This function also supports thresholding small-norm slices:
    any slice (i.e., row or column) with norm below a specified
    ``threshold`` can be left un-normalized, set to all-zeros, or
    filled with uniform non-zero values that normalize to 1.
    Note: the semantics of this function differ from
    `scipy.linalg.norm` in two ways: multi-dimensional arrays
    are supported, but matrix-norms are not.
    Parameters
    ----------
    S : np.ndarray
        The matrix to normalize
    norm : {np.inf, -np.inf, 0, float > 0, None}
        - `np.inf`  : maximum absolute value
        - `-np.inf` : mininum absolute value
        - `0`    : number of non-zeros (the support)
        - float  : corresponding l_p norm
            See `scipy.linalg.norm` for details.
        - None : no normalization is performed
    axis : int [scalar]
        Axis along which to compute the norm.
    threshold : number > 0 [optional]
        Only the columns (or rows) with norm at least ``threshold`` are
        normalized.
        By default, the threshold is determined from
        the numerical precision of ``S.dtype``.
    fill : None or bool
        If None, then columns (or rows) with norm below ``threshold``
        are left as is.
        If False, then columns (rows) with norm below ``threshold``
        are set to 0.
        If True, then columns (rows) with norm below ``threshold``
        are filled uniformly such that the corresponding norm is 1.
        .. note:: ``fill=True`` is incompatible with ``norm=0`` because
            no uniform vector exists with l0 "norm" equal to 1.
    Returns
    -------
    S_norm : np.ndarray [shape=S.shape]
        Normalized array
    Raises
    ------
    ParameterError
        If ``norm`` is not among the valid types defined above
        If ``S`` is not finite
        If ``fill=True`` and ``norm=0``
    See Also
    --------
    scipy.linalg.norm
    Notes
    -----
    This function caches at level 40.
    Examples
    --------
    >>> # Construct an example matrix
    >>> S = np.vander(np.arange(-2.0, 2.0))
    >>> S
    array([[-8.,  4., -2.,  1.],
           [-1.,  1., -1.,  1.],
           [ 0.,  0.,  0.,  1.],
           [ 1.,  1.,  1.,  1.]])
    >>> # Max (l-infinity)-normalize the columns
    >>> librosa.util.normalize(S)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # Max (l-infinity)-normalize the rows
    >>> librosa.util.normalize(S, axis=1)
    array([[-1.   ,  0.5  , -0.25 ,  0.125],
           [-1.   ,  1.   , -1.   ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 1.   ,  1.   ,  1.   ,  1.   ]])
    >>> # l1-normalize the columns
    >>> librosa.util.normalize(S, norm=1)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    >>> # l2-normalize the columns
    >>> librosa.util.normalize(S, norm=2)
    array([[-0.985,  0.943, -0.816,  0.5  ],
           [-0.123,  0.236, -0.408,  0.5  ],
           [ 0.   ,  0.   ,  0.   ,  0.5  ],
           [ 0.123,  0.236,  0.408,  0.5  ]])
    >>> # Thresholding and filling
    >>> S[:, -1] = 1e-308
    >>> S
    array([[ -8.000e+000,   4.000e+000,  -2.000e+000,
              1.000e-308],
           [ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.000e+000,   1.000e+000,   1.000e+000,
              1.000e-308]])
    >>> # By default, small-norm columns are left untouched
    >>> librosa.util.normalize(S)
    array([[ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [ -1.250e-001,   2.500e-001,  -5.000e-001,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.250e-001,   2.500e-001,   5.000e-001,
              1.000e-308]])
    >>> # Small-norm columns can be zeroed out
    >>> librosa.util.normalize(S, fill=False)
    array([[-1.   ,  1.   , -1.   ,  0.   ],
           [-0.125,  0.25 , -0.5  ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.125,  0.25 ,  0.5  ,  0.   ]])
    >>> # Or set to constant with unit-norm
    >>> librosa.util.normalize(S, fill=True)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # With an l1 norm instead of max-norm
    >>> librosa.util.normalize(S, norm=1, fill=True)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    """

    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise ParameterError(
            "threshold={} must be strictly " "positive".format(threshold)
        )

    if fill not in [None, False, True]:
        raise ParameterError("fill={} must be None or boolean".format(fill))

    if not np.all(np.isfinite(S)):
        raise ParameterError("Input must be finite")

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ParameterError("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag ** norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    elif norm is None:
        return S

    else:
        raise ParameterError("Unsupported norm: {}".format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm

def fft_frequencies(sr=22050, n_fft=2048):
    """Alternative implementation of `np.fft.fftfreq`
    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate
    n_fft : int > 0 [scalar]
        FFT window size
    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies ``(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)``
    Examples
    --------
    >>> librosa.fft_frequencies(sr=22050, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])
    """

    return np.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)


def cqt_frequencies(n_bins, fmin, bins_per_octave=12, tuning=0.0):
    """Compute the center frequencies of Constant-Q bins.
    Examples
    --------
    >>> # Get the CQT frequencies for 24 notes, starting at C2
    >>> librosa.cqt_frequencies(24, fmin=librosa.note_to_hz('C2'))
    array([  65.406,   69.296,   73.416,   77.782,   82.407,   87.307,
             92.499,   97.999,  103.826,  110.   ,  116.541,  123.471,
            130.813,  138.591,  146.832,  155.563,  164.814,  174.614,
            184.997,  195.998,  207.652,  220.   ,  233.082,  246.942])
    Parameters
    ----------
    n_bins  : int > 0 [scalar]
        Number of constant-Q bins
    fmin    : float > 0 [scalar]
        Minimum frequency
    bins_per_octave : int > 0 [scalar]
        Number of bins per octave
    tuning : float
        Deviation from A440 tuning in fractional bins
    Returns
    -------
    frequencies : np.ndarray [shape=(n_bins,)]
        Center frequency for each CQT bin
    """

    correction = 2.0 ** (float(tuning) / bins_per_octave)
    frequencies = 2.0 ** (np.arange(0, n_bins, dtype=float) / bins_per_octave)

    return correction * fmin * frequencies


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """Compute an array of acoustic frequencies tuned to the mel scale.
    The mel scale is a quasi-logarithmic function of acoustic frequency
    designed such that perceptually similar pitch intervals (e.g. octaves)
    appear equal in width over the full hearing range.
    Because the definition of the mel scale is conditioned by a finite number
    of subjective psychoaoustical experiments, several implementations coexist
    in the audio signal processing literature [#]_. By default, librosa replicates
    the behavior of the well-established MATLAB Auditory Toolbox of Slaney [#]_.
    According to this default implementation,  the conversion from Hertz to mel is
    linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
    replicates the Hidden Markov Toolkit [#]_ (HTK) according to the following formula::
        mel = 2595.0 * np.log10(1.0 + f / 700.0).
    The choice of implementation is determined by the ``htk`` keyword argument: setting
    ``htk=False`` leads to the Auditory toolbox implementation, whereas setting it ``htk=True``
    leads to the HTK implementation.
    .. [#] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
        In Proc. International Conference on Acoustics, Speech, and Signal Processing
        (ICASSP), vol. 1, pp. 217-220, 1998.
    .. [#] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
        Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.
    .. [#] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., Liu, X.,
        Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
        The HTK book, version 3.4. Cambridge University, March 2009.
    See Also
    --------
    hz_to_mel
    mel_to_hz
    librosa.feature.melspectrogram
    librosa.feature.mfcc
    Parameters
    ----------
    n_mels    : int > 0 [scalar]
        Number of mel bins.
    fmin      : float >= 0 [scalar]
        Minimum frequency (Hz).
    fmax      : float >= 0 [scalar]
        Maximum frequency (Hz).
    htk       : bool
        If True, use HTK formula to convert Hz to mel.
        Otherwise (False), use Slaney's Auditory Toolbox.
    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of ``n_mels`` frequencies in Hz which are uniformly spaced on the Mel
        axis.
    Examples
    --------
    >>> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])
    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)


def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels
    Examples
    --------
    >>> librosa.hz_to_mel(60)
    0.9
    >>> librosa.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])
    Parameters
    ----------
    frequencies   : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk           : bool
        use HTK formula instead of Slaney
    Returns
    -------
    mels        : number or np.ndarray [shape=(n,)]
        input frequencies in Mels
    See Also
    --------
    mel_to_hz
    """

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies
    Examples
    --------
    >>> librosa.mel_to_hz(3)
    200.
    >>> librosa.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])
    Parameters
    ----------
    mels          : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk           : bool
        use HTK formula instead of Slaney
    Returns
    -------
    frequencies   : np.ndarray [shape=(n,)]
        input mels in Hz
    See Also
    --------
    hz_to_mel
    """

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs
