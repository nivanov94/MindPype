# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:51:07 2019

filter.py - Defines the filter Class for MindPype

@author: ivanovn
"""

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

    """
    # these are the possible internal methods for storing the filter
    # parameters which determine how it will be executed
    implementations = ['ba', 'zpk', 'sos', 'fir']
    btypes = ['lowpass', 'highpass', 'bandpass', 'bandstop']
    ftypes = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel', 'fir']

    def __init__(self, sess, ftype, btype, implementation, crit_frqs,
                 fs, coeffs):
        """
        Constructor for the Filter class
        """
        super().__init__(MPEnums.FILTER, sess)

        self._ftype = ftype
        self._btype = btype
        self._implementation = implementation
        self._fs = fs
        self._crit_frqs = crit_frqs

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

    # API Getters
    @property
    def ftype(self):
        """
        Getter for the filter type

        Return
        -------
        The filter type, one of 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'

        Return Type
        -----------
        str
        """
        return self._ftype

    @property
    def btype(self):
        """
        Getter method for the band type used by the filter

        Return
        ------
        The band type, one of 'lowpass', 'highpass', 'bandpass', 'bandstop'

        Return Type
        -----------
        str
        """

        return self._btype

    @property
    def implementation(self):
        """
        Getter method for the filter implementation

        Returns
        -------
        The filter implementation, one of 'ba', 'zpk', 'sos'

        Return Type
        -----------
        str

        """
        return self._implementation

    @property
    def fs(self):
        """
        Getter method for the sampling frequency

        Returns
        -------
        The sampling frequency

        Return Type
        -----------
        float
        """
        return self._fs

    @property
    def crit_frqs(self):
        """
        Getter method for the critical frequencies

        Returns
        -------
        The critical frequencies

        Return Type
        -----------
        array_like of floats
        """
        return self._crit_frqs

    @property
    def coeffs(self):
        """
        Getter method for the filter coefficients

        Returns
        -------
        The filter coefficients

        Return Type
        -----------
        array_like of floats
        """
        return self._coeffs

    @classmethod
    def create_butter(cls, sess, N, Wn, btype='lowpass',
                      implementation='ba', fs=1.0):
        """
        Factory method to create a butterworth MindPype filter object

        Butterworth digital and analog filter design.

        Design an Nth-order digital or analog Butterworth filter and
        return the filter coefficients.

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
        BCIpy Filter object : Filter
            The filter object containing the filter and its parameters

        """
        coeffs = {}
        if implementation == 'ba':
            b, a = signal.butter(N, Wn, btype=btype, output=implementation,
                                 fs=fs)
            coeffs['a'] = a
            coeffs['b'] = b
        elif implementation == 'zpk':
            z, p, k = signal.butter(N, Wn, btype=btype, output=implementation,
                                    fs=fs)
            coeffs['z'] = z
            coeffs['p'] = p
            coeffs['k'] = k
        else:
            sos = signal.butter(N, Wn, btype=btype, output=implementation,
                                fs=fs)
            coeffs['sos'] = sos

        f = cls(sess, 'butter', btype, implementation, Wn, fs, coeffs)

        # add the filter to the session
        sess.add_misc_mp_obj(f)

        return f

    @classmethod
    def create_cheby1(cls, sess, N, rp, Wn, btype='lowpass',
                      implementation='ba', fs=1.0):
        """
        Factory method to create a Chebyshev Type-I MindPype filter object

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
        MindPype Filter object : Filter
            The filter object containing the filter and its parameters

        """
        coeffs = {}
        if implementation == 'ba':
            b, a = signal.cheby1(N, rp, Wn, btype=btype, output=implementation,
                                 fs=fs)
            coeffs['a'] = a
            coeffs['b'] = b
        elif implementation == 'zpk':
            z, p, k = signal.cheby1(N, rp, Wn, btype=btype,
                                    output=implementation, fs=fs)
            coeffs['z'] = z
            coeffs['p'] = p
            coeffs['k'] = k
        else:
            sos = signal.cheby1(N, rp, Wn, btype=btype, output=implementation,
                                fs=fs)
            coeffs['sos'] = sos

        f = cls(sess, 'cheby1', btype, implementation, Wn, fs, coeffs)

        # add the filter to the session
        sess.add_misc_mp_obj(f)

        return f

    @classmethod
    def create_cheby2(cls, sess, N, rs, Wn, btype='lowpass',
                      implementation='ba', fs=1.0):
        """
        Factory method to create a Chebyshev Type-II MindPype filter object

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
        MindPype Filter object : Filter
            The filter object containing the filter and its parameters

        """
        coeffs = {}
        if implementation == 'ba':
            b, a = signal.cheby2(N, rs, Wn, btype=btype, output=implementation,
                                 fs=fs)
            coeffs['a'] = a
            coeffs['b'] = b
        elif implementation == 'zpk':
            z, p, k = signal.cheby2(N, rs, Wn, btype=btype,
                                    output=implementation, fs=fs)
            coeffs['z'] = z
            coeffs['p'] = p
            coeffs['k'] = k
        else:
            sos = signal.cheby2(N, rs, Wn, btype=btype, output=implementation,
                                fs=fs)
            coeffs['sos'] = sos

        f = cls(sess, 'cheby2', btype, implementation, Wn, fs, coeffs)

        # add the filter to the session
        sess.add_misc_mp_obj(f)

        return f

    @classmethod
    def create_ellip(cls, sess, N, rp, rs, Wn, btype='lowpass',
                     implementation='ba', fs=1.0):
        """
        Factory method to create a Elliptic MindPype filter object

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
        MindPype Filter object : Filter
            The filter object containing the filter and its parameters

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
        sess.add_misc_mp_obj(f)

        return f

    @classmethod
    def create_bessel(cls, sess, N, Wn, btype='lowpass',
                      implementation='ba', norm='phase', fs=1.0):
        """
        Factory method to create a Bessel MindPype filter object

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
        MindPype Filter object : Filter
            The filter object containing the filter and its parameters


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
        sess.add_misc_mp_obj(f)

        return f

    @classmethod
    def create_fir(cls,
                   sess,
                   fs,
                   low_freq=None,
                   high_freq=None,
                   filter_length="auto",
                   l_trans_bandwidth="auto",
                   h_trans_bandwidth="auto",
                   method="fir",
                   iir_params=None,
                   phase="zero",
                   fir_window="hamming",
                   fir_design="firwin"):
        """
        Factory method to create a FIR MindPype filter object. Creates a
        Scipy.signal.firwin object and stores it in the filter object.

        .. note::
            The FIR is based on the Scipy firwin class, visit the
            `Scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html>`_
            for more information on the parameters.

        Parameters
        ----------

        sess : MindPype Session object
            The session object to which the filter will be added

        Other Parameters are the same as the MNE create_filter method, see the
        `MNE documentation <https://mne.tools/stable/generated/mne.filter.create_filter.html>`_
        for more information on the parameters.

        Returns
        ------
        MindPype Filter object : Filter
            The filter object containing the filter and its parameters

        Raises
        ------
        ValueError
            If any value in cutoff is less than or equal to 0 or greater than
            or equal to fs/2, if the values in cutoff are not strictly
            monotonically increasing.
        """
        coeffs = {}

        coeffs['fir'] = mne.filter.create_filter(None, fs, low_freq, high_freq,
                                                 filter_length,
                                                 l_trans_bandwidth,
                                                 h_trans_bandwidth,
                                                 method, None, phase,
                                                 fir_window, fir_design)
        coeffs['phase'] = phase
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
        sess.add_misc_mp_obj(f)

        return f
