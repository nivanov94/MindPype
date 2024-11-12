from .core import MPBase, MPEnums
from scipy import signal
import mne


class Filter(MPBase):
    """
    A filter that can be used by different MindPype kernels

    Attributes
    ----------
    ftype : str, default 'butter'
        The type of filter. Can be one of 'butter', 'cheby1', 'cheby2',
        'ellip', 'bessel'
    btype : str, default 'lowpass'
        The type of filter. Can be one of 'lowpass', 'highpass',
        'bandpass', 'bandstop'
    implementation : str, default 'ba'
        The type of filter. Can be one of 'ba', 'zpk', 'sos'
    fs : float, default 1.0
        The sampling frequency of the filter
    crit_frqs : array_like of floats
        The critical frequencies of the filter. For lowpass and highpass
        filters, Wn is a scalar; for bandpass and bandstop filters, Wn is
        a length-2 sequence.
    coeffs : array_like of floats
        The filter coefficients. The coefficients depend on the filter
        type and implementation. See scipy.signal documentation for
        more details.

    .. note::
        This class uses the scipy module :module:`signal <scipy:scipy.signal>`
        to create IIR filters and the mne module :module:`filter <mne:mne.filter>`
        to create FIR filters. See the factory methods for more information on the 
        filter creation.
    """
    # these are the possible internal methods for storing the filter
    # parameters which determine how it will be executed
    implementations = ['ba', 'zpk', 'sos', 'fir']
    btypes = ['lowpass', 'highpass', 'bandpass', 'bandstop']
    ftypes = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel', 'fir']

    def __init__(
        self, sess, ftype, btype, implementation,
        crit_frqs, fs, coeffs
    ):
        """ Init. """
        super().__init__(MPEnums.FILTER, sess)

        self.ftype = ftype
        self.btype = btype
        self.implementation = implementation
        self.fs = fs
        self.crit_frqs = crit_frqs
        self._coeffs = coeffs

    def __str__(self):
        """
        Returns a string representation of the filter

        Returns
        -------
        str
            A string representation of the filter
        """
        return (("MindPype Filter with following" +
                 "attributes:\nFilter Type: {}\nBand Type: {}\n" +
                 "Implementation: {}\nSampling Frequency: {}\n" +
                 "Critical Frequencies: {}").format(self.ftype, self.btype,
                                                    self.implementation,
                                                    self.fs, self.crit_frqs))

    @classmethod
    def create_butter(
        cls, sess, N, Wn, btype='lowpass', implementation='ba', fs=1.0
    ):
        """
        Factory method to create a butterworth MindPype filter object
        using the scipy.signal.butter method. See the scipy documentation
        :method:`butter <scipy:scipy.signal.butter>`
        for more details and parameter details.

        Parameters
        ----------
        N : int
            The order of the filter.
        Wn : array_like
            The critical frequency or frequencies. For lowpass and highpass
            filters, Wn is a scalar; for bandpass and bandstop filters, Wn
            is a length-2 sequence.
            For a Butterworth filter, this is the point at which the gain
            drops to 1/sqrt(2) that of the passband (the "-3 dB point").
            For digital filters, if fs is not specified, Wn units are
            normalized from 0 to 1, where 1 is the Nyquist frequency
            (Wn is thus in half cycles / sample and defined as 2*critical
            frequencies / fs). If fs is specified, Wn is in the same units
            as fs. For analog filters, Wn is an angular frequency (e.g. rad/s).
        btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'},
        default: lowpass
            The type of filter. Default is 'lowpass'.
        output : {'ba', 'zpk', 'sos'}, default: 'ba'
            Type of output: numerator/denominator ('ba'), pole-zero ('zpk'), or
            second-order sections ('sos'). Default is 'ba' for backwards
            compatibility, but 'sos' should be used for general-purpose
            filtering.
        fs : float, default: 1.0
            The sampling frequency of the digital system.

        Return
        ------
        Filter
            The MindPype filter object containing the filter and its parameters

        """
        coeffs = {}
        if implementation == 'ba':
            b, a = signal.butter(N, Wn, btype=btype, 
                                 output=implementation, fs=fs)
            coeffs['a'] = a
            coeffs['b'] = b
        elif implementation == 'zpk':
            z, p, k = signal.butter(N, Wn, btype=btype, 
                                    output=implementation, fs=fs)
            coeffs['z'] = z
            coeffs['p'] = p
            coeffs['k'] = k
        else:
            sos = signal.butter(N, Wn, btype=btype, 
                                output=implementation, fs=fs)
            coeffs['sos'] = sos

        f = cls(sess, 'butter', btype, implementation, Wn, fs, coeffs)

        # add the filter to the session
        sess.add_to_session(f)

        return f

    @classmethod
    def create_cheby1(
        cls, sess, N, rp, Wn, btype='lowpass', implementation='ba', fs=1.0
    ):
        """
        Factory method to create a Chebyshev Type-I MindPype filter object
        using the scipy.signal.cheby1 method. See the scipy documentation
        :method:`cheby1 <scipy:scipy.signal.cheby1>`
        for more details and parameter details.

        Parameters
        ----------
        sess : Session
            Session where the filter object will exist
        N : int
            The order of the filter.
        rp : float
            The maximum ripple allowed below unity gain in the passband.
            Specified in decibels, as a positive number.
        Wn : array_like
            A scalar or length-2 sequence giving the critical frequencies.
            For Type I filters, this is the point in the transition band
            at which the gain first drops below -rp.
            For digital filters, Wn are in the same units as fs. By
            default, fs is 2 half-cycles/sample, so these are normalized
            from 0 to 1, where 1 is the Nyquist frequency. (Wn is thus in
            half-cycles / sample.)
            For analog filters, Wn is an angular frequency (e.g., rad/s).
        btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, default:
        lowpass
            The type of filter. Default is 'lowpass'.
        output : {'ba', 'zpk', 'sos'}, default: 'ba'
            Type of output: numerator/denominator ('ba'), pole-zero ('zpk'),
            or second-order sections ('sos'). Default is 'ba' for backwards
            compatibility, but 'sos' should be used for general-purpose
            filtering.
        fs : float, default: 1.0
            The sampling frequency of the digital system.

        Return
        ------
        Filter
            The MindPype filter object containing the filter and its parameters

        """
        coeffs = {}
        if implementation == 'ba':
            b, a = signal.cheby1(N, rp, Wn, btype=btype, 
                                 output=implementation, fs=fs)
            coeffs['a'] = a
            coeffs['b'] = b
        elif implementation == 'zpk':
            z, p, k = signal.cheby1(N, rp, Wn, btype=btype,
                                    output=implementation, fs=fs)
            coeffs['z'] = z
            coeffs['p'] = p
            coeffs['k'] = k
        else:
            sos = signal.cheby1(N, rp, Wn, btype=btype, 
                                output=implementation, fs=fs)
            coeffs['sos'] = sos

        f = cls(sess, 'cheby1', btype, implementation, Wn, fs, coeffs)

        # add the filter to the session
        sess.add_to_session(f)

        return f

    @classmethod
    def create_cheby2(cls, sess, N, rs, Wn, btype='lowpass',
                      implementation='ba', fs=1.0):
        """
        Factory method to create a Chebyshev Type-II MindPype filter object
        using the scipy.signal.cheby2 method. See the scipy documentation
        :method:`cheby2 <scipy:scipy.signal.cheby2>`
        for more details and parameter details.

        Parameters
        ----------
        sess : Session
            Session where the filter object will exist
        N : int
            The order of the filter.
        rs : float
            The minimum attenuation required in the stop band.
            Specified in decibels, as a positive number.
        Wn : array_like
            A scalar or length-2 sequence giving the critical frequencies.
            For Type II filters, this is the point in the transition band at
            which the gain first reaches -rs.
            For digital filters, Wn are in the same units as fs. By default,
            fs is 2 half-cycles/sample, so these are normalized from 0 to 1,
            where 1 is the Nyquist frequency.
            (Wn is thus in half-cycles / sample.)
            For analog filters, Wn is an angular frequency (e.g., rad/s).
        btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'},
        default: 'lowpass'
            The type of filter. Default is 'lowpass'.
        output : {'ba', 'zpk', 'sos'}, default: 'ba'
            Type of output: numerator/denominator ('ba'), pole-zero ('zpk'),
            or second-order sections ('sos'). Default is 'ba' for backwards
            compatibility, but 'sos' should be used for general-purpose
            filtering.
        fs : float, default: 1.0
            The sampling frequency of the digital system.

        Return
        ------
        Filter
            The MindPype filter object containing the filter and its parameters

        """
        coeffs = {}
        if implementation == 'ba':
            b, a = signal.cheby2(N, rs, Wn, btype=btype,
                                 output=implementation, fs=fs)
            coeffs['a'] = a
            coeffs['b'] = b
        elif implementation == 'zpk':
            z, p, k = signal.cheby2(N, rs, Wn, btype=btype,
                                    output=implementation, fs=fs)
            coeffs['z'] = z
            coeffs['p'] = p
            coeffs['k'] = k
        else:
            sos = signal.cheby2(N, rs, Wn, btype=btype, 
                                output=implementation, fs=fs)
            coeffs['sos'] = sos

        f = cls(sess, 'cheby2', btype, implementation, Wn, fs, coeffs)

        # add the filter to the session
        sess.add_to_session(f)

        return f

    @classmethod
    def create_ellip(cls, sess, N, rp, rs, Wn, btype='lowpass',
                     implementation='ba', fs=1.0):
        """
        Factory method to create a Elliptic MindPype filter object
        using the scipy.signal.ellip method. See the scipy documentation
        :method:`ellip <scipy:scipy.signal.ellip>`
        for more details and parameter details.

        Parameters
        ----------
        N : int
            The order of the filter.
        rp : float
            The maximum ripple allowed below unity gain in the passband.
            Specified in decibels, as a positive number.
        rs : float
            The minimum attenuation required in the stop band. Specified in
            decibels, as a positive number.
        Wn : array_like
            A scalar or length-2 sequence giving the critical frequencies. For
            elliptic filters, this is the point in the transition band at
            which the gain first drops below -rp.
            For digital filters, Wn are in the same units as fs. By default,
            fs is 2 half-cycles/sample, so these are normalized from 0 to 1,
            where 1 is the Nyquist frequency. (Wn is thus in
            half-cycles / sample.)
            For analog filters, Wn is an angular frequency (e.g., rad/s).
        btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
            The type of filter. Default is 'lowpass'.
        analog : bool, optional
            When True, return an analog filter, otherwise a digital filter
            is returned.
        output : {'ba', 'zpk', 'sos'}, optional
            Type of output: numerator/denominator ('ba'), pole-zero ('zpk'),
            or second-order sections ('sos'). Default is 'ba' for backwards
            compatibility, but 'sos' should be used for general-purpose
            filtering.
        fs : float, optional
            The sampling frequency of the digital system.

        Return
        ------
        Filter
            The MindPype filter object containing the filter and its parameters

        """
        coeffs = {}
        if implementation == 'ba':
            b, a = signal.ellip(N, rp, rs, Wn,
                                btype=btype, output=implementation, fs=fs)
            coeffs['a'] = a
            coeffs['b'] = b
        elif implementation == 'zpk':
            z, p, k = signal.ellip(N, rp, rs, Wn,
                                   btype=btype, output=implementation, fs=fs)
            coeffs['z'] = z
            coeffs['p'] = p
            coeffs['k'] = k
        else:
            sos = signal.ellip(N, rp, rs, Wn,
                               btype=btype, output=implementation, fs=fs)
            coeffs['sos'] = sos

        f = cls(sess, 'ellip', btype, implementation, Wn, fs, coeffs)

        # add the filter to the session
        sess.add_to_session(f)

        return f

    @classmethod
    def create_bessel(cls, sess, N, Wn, btype='lowpass',
                      implementation='ba', norm='phase', fs=1.0):
        """
        Factory method to create a Bessel MindPype filter object
        using the scipy.signal.bessel method. See the scipy documentation
        :method:`bessel <scipy:scipy.signal.bessel>`
        for more details and parameter details.

        Parameters
        ----------
        N : int
            The order of the filter.
        Wn : array_like
            A scalar or length-2 sequence giving the critical frequencies
            (defined by the norm parameter). For analog filters, Wn is
            an angular frequency (e.g., rad/s).
            For digital filters, Wn are in the same units as fs. By default,
            fs is 2 half-cycles/sample, so these are normalized from 0 to 1,
            where 1 is the Nyquist frequency. (Wn is thus in
            half-cycles / sample.)
        btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
            The type of filter. Default is 'lowpass'.
        analog : bool, optional
            When True, return an analog filter, otherwise a digital filter
            is returned. (See Notes.)
        output : {'ba', 'zpk', 'sos'}, optional
            Type of output: numerator/denominator ('ba'), pole-zero ('zpk'),
            or second-order sections ('sos'). Default is 'ba'.
        norm : {'phase', 'delay', 'mag'}, optional
            Critical frequency normalization:
                phase
                    The filter is normalized such that the phase response
                    reaches its midpoint at angular (e.g. rad/s) frequency Wn.
                    This happens for both low-pass and high-pass filters, so
                    this is the "phase-matched" case.
                    The magnitude response asymptotes are the
                    same as a Butterworth filter of the same order with a
                    cutoff of Wn.  This is the default, and matches MATLAB's
                    implementation.
                delay
                    The filter is normalized such that the group delay in
                    the passband is 1/Wn (e.g., seconds). This is the "natural"
                    type obtained by solving Bessel polynomials.
                mag
                    The filter is normalized such that the gain magnitude is
                    -3 dB at angular frequency Wn.
        fs : float, optional
            The sampling frequency of the digital system.

        Return
        ------
        Filter
            The MindPype filter object containing the filter and its parameters


        """
        coeffs = {}
        if implementation == 'ba':
            b, a = signal.bessel(N, Wn, btype=btype, output=implementation,
                                 fs=fs)
            coeffs['a'] = a
            coeffs['b'] = b
        elif implementation == 'zpk':
            z, p, k = signal.bessel(N, Wn, btype=btype, output=implementation,
                                    fs=fs)
            coeffs['z'] = z
            coeffs['p'] = p
            coeffs['k'] = k
        else:
            sos = signal.bessel(N, Wn, btype=btype, output=implementation,
                                fs=fs)
            coeffs['sos'] = sos

        f = cls(sess, 'bessel', btype, implementation, Wn, fs, coeffs)

        # add the filter to the session
        sess.add_to_session(f)

        return f

    @classmethod
    def create_fir(
        cls,
        sess,
        fs,
        low_freq=None,
        high_freq=None,
        filter_length="auto",
        l_trans_bandwidth="auto",
        h_trans_bandwidth="auto",
        fir_window="hamming",
        fir_design="firwin"
    ):
        """
        Factory method to create a FIR MindPype filter object
        using the mne.filter.create_filter method. See the mne documentation
        :method:`create_filter <mne:mne.filter.create_filter>`
        for more details and parameter details.

        Parameters
        ----------
        sess : Session 
            The session object to which the filter will be added
        fs : float
            The sampling frequency of the signal
        low_freq : float, default None
            The low cutoff frequency in Hz. If None, a high-pass filter
            is created.
        high_freq : float, default None
            The high cutoff frequency in Hz. If None, a low-pass filter
            is created.
        filter_length : str or int, default 'auto'
            Length of the FIR filter to use (if applicable). Can be 'auto'
            (default) to use the minimum good length for the given filter
            type, or an integer to specify the length directly.
        l_trans_bandwidth : str or float, default 'auto'
            Width of the transition band at the low cut-off frequency in Hz
            (high-pass and band-stop filters) or at the high cut-off frequency
            in Hz (low-pass and band-pass filters). Can be 'auto' (default)
            to use a multiple of the cutoff frequency, or a float to specify
            the exact transition bandwidth.
        h_trans_bandwidth : str or float, default 'auto'
            Width of the transition band at the high cut-off frequency in Hz
            (low-pass and band-stop filters) or at the low cut-off frequency
            in Hz (high-pass and band-pass filters). Can be 'auto' (default)
            to use a multiple of the cutoff frequency, or a float to specify
            the exact transition bandwidth.
        fir_window : str, default 'hamming'
            The window to use in FIR design. See mne documentation for
            available windows.
        fir_design : str, default 'firwin'
            The method to use for FIR design. See mne documentation for
            available FIR design methods.

        Returns
        ------
        Filter
            The MindPype filter object containing the filter and its parameters
        """
        coeffs = {
            'fir' : mne.filter.create_filter(None, fs, low_freq, 
                                             high_freq,
                                             filter_length,
                                             l_trans_bandwidth,
                                             h_trans_bandwidth,
                                             "fir", None, 'zero',
                                             fir_window, fir_design),
            'phase' : 'zero'
        }

        if low_freq is None and high_freq is not None:
            btype = 'lowpass'
        elif low_freq is not None and high_freq is None:
            btype = 'highpass'
        elif ((low_freq is not None and high_freq is not None) and
              (low_freq < high_freq)):
            btype = 'bandpass'
        elif ((low_freq is not None and high_freq is not None) and
              (low_freq > high_freq)):
            btype = 'bandstop'

        f = cls(sess, 'fir', btype, 'fir', crit_frqs=[low_freq, high_freq],
                fs=fs, coeffs=coeffs)

        # add the filter to the session
        sess.add_to_session(f)

        return f
