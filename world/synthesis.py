#3rd party imports
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
#from scipy.io.wavfile import write

#build-in imports
from decimal import Decimal, ROUND_HALF_UP
import sys
#from .fftfilt import fftfilt

import cython
#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)

#import numba

@cython.locals(i=cython.int)
#@numba.jit()
def synthesis(source_object, filter_object):
    '''
    Waveform synthesis from the estimated parameters
    y = Synthesis(source_object, filter_object)
    
    Args:
        source_object - WORLD dict containing vuv/f0/temporal_positions/aperiodicity
            temporal_positions :: [float] - Times of input's data-points [sec], e.g. [0.000, 0.005, 0.010, 0.015, ...]
        filter_object - WORLD dict containing fs/spectrogram
    Returns:
        y - synthesized waveform    
    '''
    # NOTE: Intended to be called by `World.decode` internally.
    # Actually, source_object === filter_object in `World.decode`.

    # Configs
    default_f0: int = 500 # f0 value in unvoiced region

    # Args
    vuv                = source_object['vuv']
    f0                 = source_object['f0']
    temporal_positions = source_object['temporal_positions']
    fs                 = filter_object['fs']
    spectrogram        = filter_object['spectrogram']

    # Times of output's data-point [sec]
    ## e.g. [0, 1/fs, 2/fs, ..., T]
    time_axis = np.arange(temporal_positions[0], temporal_positions[-1] + 1 / fs, 1 / fs)

    # y :: (T,) - Output
    y = np.zeros(len(time_axis))
    y_length = len(y)

    # Pulse position base on fo contour & VUV
    pulse_locations, pulse_locations_index, pulse_locations_time_shift, interpolated_vuv = time_base_generation(temporal_positions, f0, fs, vuv, time_axis, default_f0)
    # temporal_position_index :: float[] - Pulse positions as temporal_positions's continuous indice, 1 <= temporal_position_index <= len(temporal_positions)
    temporal_position_index = interp1d(temporal_positions, np.arange(1, len(temporal_positions) + 1), fill_value='extrapolate')(pulse_locations)
    temporal_position_index = np.maximum(1, np.minimum(len(temporal_positions), temporal_position_index))

    fft_size: int = (spectrogram.shape[0] - 1) * 2
    # base_index -   e.g. fft_size=512 -> [-255, -254, ..., +255, +256]
    base_index = np.arange(-fft_size // 2 + 1, fft_size // 2 + 1)
    # latter_index - e.g. fft_size=512 -> [+257, +258, ..., +511, +512]
    latter_index = np.arange(fft_size // 2 + 1, fft_size + 1)

    # NOTE: periodic + aperiodic = 1
    amplitude_aperiodic = source_object['aperiodicity'] ** 2
    amplitude_periodic = np.maximum(0.001, (1 - amplitude_aperiodic))

    dc_remover_base = signal.hanning(fft_size + 2)[1:-1]
    dc_remover_base = dc_remover_base / np.sum(dc_remover_base)

    # coefficient :: float - 2π*
    coefficient = 2.0 * np.pi * fs / fft_size
    
    for i in range(len(pulse_locations_index)):
        # NOTE: spec/periodic/aperiodic -> an interpolated frame :: (Freq,)
        spectrum_slice, periodic_slice, aperiodic_slice = \
            get_spectral_parameters(temporal_positions, temporal_position_index[i], spectrogram, amplitude_periodic, amplitude_aperiodic, pulse_locations[i])

        # inter-pulse length, time_axis scale
        noise_size: int = pulse_locations_index[min(len(pulse_locations_index) - 1, i + 1)] - pulse_locations_index[i]

        # 1 <= `pulse_locations_index[i] + base_index` <= y_length
        # [t_p-F, ... t_p-1, t_p, t_p+1, ..., t_p+F]
        output_buffer_indice = np.maximum(1, np.minimum(y_length, pulse_locations_index[i] + base_index))
        
        if interpolated_vuv[pulse_locations_index[i] - 1] >= 0.5 and aperiodic_slice[0] <= 0.999:
            # NOTE: (maybe) periodic components
            ## NOTE: (maybe) Generate filtered harmonics
            response = get_periodic_response(spectrum_slice, periodic_slice, fft_size, latter_index, pulse_locations_time_shift[i], coefficient)
            dc_remover = dc_remover_base * -np.sum(response)
            periodic_response = response + dc_remover
            ## NOTE: Add colored harmonics to Output
            y[output_buffer_indice.astype(int) - 1] += periodic_response * np.sqrt(max(1, noise_size))

        # NOTE: aperiodic components
            ## NOTE: weighting
            tmp_aperiodic_spectrum = spectrum_slice * aperiodic_slice
        else:
            ## NOTE: (maybe) weighting with 1 (>=0.999)
            tmp_aperiodic_spectrum = spectrum_slice
        ## NOTE: Spectrum flooring
        tmp_aperiodic_spectrum[tmp_aperiodic_spectrum == 0] = sys.float_info.epsilon
        ## NOTE: Signal Synthesis
        aperiodic_response = get_aperiodic_response(tmp_aperiodic_spectrum, fft_size, latter_index, noise_size)
        ## Add colored noise to Output
        y[output_buffer_indice.astype(int) - 1] += aperiodic_response
    return y

#####################################################

def get_aperiodic_response(tmp_aperiodic_spectrum, fft_size, latter_index, noise_size):
    """Generate an aperiodic signal by given spectrum + sampled noise."""

    aperiodic_spectrum = np.r_[tmp_aperiodic_spectrum, tmp_aperiodic_spectrum[-2: 0: -1]]
    # NOTE: spec-to-ceps
    tmp_cepstrum = np.fft.fft((np.log(np.abs(aperiodic_spectrum)) / 2)).real
    # NOTE: Liftering or something...? zeros -> cep*2 for some elements except for [0] & *1 for [0]
    tmp_complex_cepstrum = np.zeros(fft_size)
    tmp_complex_cepstrum[latter_index.astype(int) - 1] = tmp_cepstrum[latter_index.astype(int) - 1] * 2
    tmp_complex_cepstrum[0] = tmp_cepstrum[0]
    # NOTE: ceps-to-spec
    response = np.fft.fftshift(np.fft.ifft(np.exp(np.fft.ifft(tmp_complex_cepstrum))).real)
    noise_input = np.random.randn(max(3, noise_size))
    # noise_input = np.zeros(max(3, noise_size)) + 0.1
    # NOTE: Apply `response` IIR filter to the white noise
    response = fftfilt(noise_input - np.mean(noise_input), response)
    return response

#######################################################

def get_periodic_response(spectrum_slice, periodic_slice, fft_size, latter_index, fractionsl_time_shift: float, coefficient):
    """Generate FIR filter coefficients.
    
    Args:
        spectrum_slice :: maybe (Freq,) -
        periodic_slice :: maybe (Freq,) -
        fft_size
        latter_index                    - e.g. fft_size=512 -> [+257, +258, ..., +511, +512]
        fractionsl_time_shift           - Time shift for exact pulse position [sec]
        coefficient                     - e.g. 2.0 * np.pi * fs / fft_size
    Returns:
        response - FIR filtered waveform
    """
    # Harmonic amplitudes
    tmp_spectrum_slice = spectrum_slice * periodic_slice
    tmp_spectrum_slice[tmp_spectrum_slice == 0] = sys.float_info.epsilon

    periodic_spectrum = np.r_[tmp_spectrum_slice, tmp_spectrum_slice[-2: 0: -1]]
    # NOTE: Same as aperiodic
    # NOTE: Liftering or something...? zeros -> cep*2 for some elements except for [0] & *1 for [0]
    tmp_cepstrum = np.fft.fft(np.log(np.abs(periodic_spectrum)) / 2).real
    # [#0, 0, 0, ..., 0, 2*#half, 2*#{half+1}, ...]
    tmp_complex_cepstrum = np.zeros(fft_size)
    tmp_complex_cepstrum[latter_index.astype(int) - 1] = tmp_cepstrum[latter_index.astype(int) - 1] * 2
    tmp_complex_cepstrum[0] = tmp_cepstrum[0]

    # (maybe) Minimum-phase filter's transfer function H(ω)
    spectrum = np.exp(np.fft.ifft(tmp_complex_cepstrum))
    spectrum = spectrum[0: int(fft_size / 2) + 1]

    # np.exp(-1j * 2π* * fractionsl_time_shift * [0, 1, ..., nfft/2])
    # np.exp(-1j * 2π*fs/nfft * sft * [0, 1, ..., nfft/2]) -> np.exp([0, -2πj*fs*1/nfft*sft, ..., -2πj*fs*{nfft-1}/nfft*sft, -2πj*fs/2*sft])
    spectrum *= np.exp(-1j * coefficient * fractionsl_time_shift * np.arange(int(fft_size / 2) + 1))
    spectrum = np.r_[spectrum, spectrum[-2: 0: -1].conj()]

    # Freq-to-Time
    response = np.fft.fftshift(np.fft.ifft(spectrum).real)

    return response

#######################################################

def time_base_generation(temporal_positions, f0, fs, vuv, signal_time, default_f0: int):
    """
    Args:
        temporal_positions :: [float] - Times of input's  data-points [sec], e.g. [0.000, 0.005, 0.010, ...]
        f0                            - fo series [?/?],           same time scale as `temporal_positions`
        fs                            -
        vuv                           - Voice/UnVoiced series [?], same time scale as `temporal_positions`
        signal_time                   - Times of output's data-points [sec], e.g. [0.0, 1/fs,  2/fs, ...]
        default_f0                    - f0 set to unvoiced region
    Returns:
        pulse_locations            :: float[] - Times just after which phase jumped over 0, signal_time's scale, e.g. [30/fs, 160/fs, 300/fs, ...]
        pulse_locations_index      ::   int[] - Indice of pulses in output series,          signal_time's scale, e.g. [30,    160,    300,    ...]
        pulse_locations_time_shift :: float[] - Time shifts for exact pulse position [sec], signal_time's scale, e.g. [0.01,  0.02,   0.00,   ...]
        vuv_interpolated           ::  bool[] - Voiced/Unvoiced flag series,                signal_time's scale

    """
    vuv_interpolated = interp1d(temporal_positions, vuv, kind='linear', fill_value='extrapolate')(signal_time)
    vuv_interpolated = vuv_interpolated > 0.5

    # f0_interpolated - fo contour [rot/sec?], UV region is set to `default_f0`
    f0_interpolated_raw = interp1d(temporal_positions, f0, kind='linear', fill_value='extrapolate')(signal_time)
    f0_interpolated = f0_interpolated_raw * vuv_interpolated
    f0_interpolated[f0_interpolated == 0] = f0_interpolated[f0_interpolated == 0] + default_f0

    # wrap_phase - Wrapped phases at each data-points
    total_phase = np.cumsum(2 * np.pi * f0_interpolated / fs)
    wrap_phase = np.remainder(total_phase, 2 * np.pi)

    # pulse_locations :: [float] - List of times just after which phase jumped over 0 (loc=0.1 if 't=0.1/p=1.99pi' & 't=0.11/p=0.02pi')
    pulse_locations = (signal_time[:-1])[np.abs(np.diff(wrap_phase)) > np.pi]
    assert (len(pulse_locations)) > 0
    # pulse_locations_index :: [int] - Indice of pulses in output series
    pulse_locations_index = np.array([int(Decimal(elm * fs).quantize(0, ROUND_HALF_UP)) for elm in pulse_locations]) + 1

    # [wrap_phase]
    #   ・1.99
    #               ->
    #      ・0.04         ・0.04   
    #                   ・-0.01  
    #   t-1 t             t-0.8
    y1 = wrap_phase[pulse_locations_index - 1] -2.0 * np.pi
    y2 = wrap_phase[pulse_locations_index]
    x = -y1 / (y2 - y1)

    pulse_locations_time_shift = x / fs

    return pulse_locations, pulse_locations_index, pulse_locations_time_shift, vuv_interpolated


#####################################################
def get_spectral_parameters(temporal_positions, temporal_position_index: float, spectrogram, amplitude_periodic, amplitude_random, pulse_locations: float):
    """(maybe) Get slice of spectrogram/periodicity/aperiodicity

    Args:
        temporal_positions      :: [float]             - Times of frame data-points [sec]
        temporal_position_index                        - Continuous index of temporal_positions at pulse timing
        spectrogram             :: maybe (Freq, Frame) - Spectrogram
        amplitude_periodic      :: maybe (Freq, Frame) - Periodicity
        amplitude_random        :: maybe (Freq, Frame) - Aperiodicity
        pulse_locations                                - The time just after which phase jumped over 0, e.g. `160/fs`
    Returns:
        spectrum_slice  :: maybe (Freq,) - spectrum, an interpolated frame of spectrogram
        periodic_slice  :: maybe (Freq,) - non-ap,   an interpolated frame of amplitude_periodic
        aperiodic_slice :: maybe (Freq,) - ap,       an interpolated frame of amplitude_random
    """
    floor_index = int(np.floor(temporal_position_index)) - 1
    ceil_index  = int(np.ceil(temporal_position_index)) - 1
    t_f1 = temporal_positions[floor_index] # Time of frame1
    t_f2 = temporal_positions[ceil_index]  # Time of frame2

    # t_f1 <= pulse_locations <= t_f2
    t_p = max(t_f1, min(t_f2, pulse_locations))

    if t_f1 == t_f2:
        # Exact frame
        spectrum_slice  =        spectrogram[:, floor_index]
        periodic_slice  = amplitude_periodic[:, floor_index]
        aperiodic_slice =   amplitude_random[:, floor_index]
    else:
        # Spectrum interpolation
        b = (t_p - t_f1) / (t_f2 - t_f1)
        assert 0 <= b <= 1
        a = 1 - b
        spectrum_slice  = a *        spectrogram[:, floor_index] + b *        spectrogram[:, ceil_index]
        periodic_slice  = a * amplitude_periodic[:, floor_index] + b * amplitude_periodic[:, ceil_index]
        aperiodic_slice = a *   amplitude_random[:, floor_index] + b *   amplitude_random[:, ceil_index]
    
    return spectrum_slice, periodic_slice, aperiodic_slice

###############################
def nextpow2(x):
    """Return the first integer N such that 2**N >= abs(x)"""

    return np.ceil(np.log2(abs(x)))

######################################
def fftfilt(b, x, *n):
    """
    Apply FIR filter to the signal using the overlap-add method.
    If the FFT length is not specified, it and the overlap-add block length are selected so as to minimize the computational cost of the filtering operation.

    Args:
        b - FIR filter's coefficients
        x - the signal
        n - FFT length
    Returns:

    """

    N_x = len(x)
    N_b = len(b)

    #### Determine the FFT length `N_fft` #########################################
    if len(n):
        # Use the specified FFT length (rounded up to the nearest power of 2), provided that it is no less than the filter length:
        n = n[0]
        if n != int(n) or n <= 0:
            raise ValueError('n must be a nonnegative integer')
        if n < N_b:
            n = N_b
        N_fft = 2**nextpow2(n)
    else:
        if N_x > N_b:

            # When the filter length is smaller than the signal,
            # choose the FFT length and block size that minimize the
            # FLOPS cost. Since the cost for a length-N FFT is
            # (N/2)*log2(N) and the filtering operation of each block
            # involves 2 FFT operations and N multiplications, the
            # cost of the overlap-add method for 1 length-N block is
            # N*(1+log2(N)). For the sake of efficiency, only FFT
            # lengths that are powers of 2 are considered:
            N = 2**np.arange(np.ceil(np.log2(N_b)),np.floor(np.log2(N_x)))
            cost = np.ceil(N_x/(N-N_b+1))*N*(np.log2(N)+1)
            assert len(cost) > 0
            N_fft = N[np.argmin(cost)]
        else:
            # When the filter length is at least as long as the signal,
            # filter the signal using a single block:
            N_fft = 2**nextpow2(N_b+N_x-1)
    N_fft = int(N_fft)
    #### /Determine the FFT length `N_fft` ########################################

    # Compute the block length:
    L = int(N_fft - N_b + 1)

    # Compute the transform of the filter:
    H = np.fft.fft(b,N_fft)

    y = np.zeros(N_x,float)
    i = 0
    while i <= N_x:
        il = min([i+L,N_x])
        k = min([i+N_fft,N_x])
        yt = np.fft.ifft(np.fft.fft(x[i:il],N_fft)*H,N_fft).real # Overlap..
        y[i:k] = y[i:k] + yt[:k-i]            # and add
        i += L
    return y
