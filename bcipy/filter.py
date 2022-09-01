# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:51:07 2019

filter.py - Defines the filter Class for BCIP

@author: ivanovn
"""

from .bcipy_core import BCIP, BcipEnums
from scipy import signal

class Filter(BCIP):
    """
    A filter that can be used by different BCIP kernels
    
    """
    
    # these are the possible internal methods for storing the filter 
    # parameters which determine how it will be executed
    implementations = ['ba', 'zpk', 'sos']
    btypes = ['lowpass','highpass','bandpass','bandstop']
    ftypes = ['butter','cheby1','cheby2','ellip','bessel']
    
    def __init__(self,sess,ftype,btype,implementation,crit_frqs,fs,coeffs):
        """
        Create a new filter object
        """
        super().__init__(BcipEnums.FILTER,sess)
        
        self._ftype = ftype
        self._btype = btype
        self._implementation = implementation
        self._fs = fs
        self._crit_frqs = crit_frqs
        
        self._coeffs = coeffs
        
    def __str__(self):
        return "BCIP {} Filter with following" + \
               "attributes:\nFilter Type: {}\nBand Type: {}\n" + \
               "Implementation: {}\nSampling Frequency: {}\n" + \
               "Critical Frequencies: {}".format(self.ftype,self.btype,
                                                 self.implementation,
                                                 self.fs, self.crit_frqs)
        
    # API Getters
    @property
    def ftype(self):
        return self._ftype
    
    @property
    def btype(self):
        return self._btype
    
    @property
    def implementation(self):
        return self._implementation
    
    @property
    def fs(self):
        return self._fs
    
    @property
    def crit_frqs(self):
        return self._crit_frqs
        
    @property
    def coeffs(self):
        return self._coeffs
    
    @classmethod
    def create_butter(cls,sess,N,Wn,btype='lowpass',implementation='ba',fs=1.0):
        """
        Factory method to create a butterworth BCIP filter object

        Butterworth digital and analog filter design.

        Design an Nth-order digital or analog Butterworth filter and return the filter coefficients.

        Parameters
        ----------
        N : int
            - The order of the filter.
        
        Wn : array_like
            - The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for bandpass and bandstop filters, Wn is a length-2 sequence.
            - For a Butterworth filter, this is the point at which the gain drops to 1/sqrt(2) that of the passband (the "-3 dB point").
            - For digital filters, if fs is not specified, Wn units are normalized from 0 to 1, where 1 is the Nyquist frequency (Wn is thus in half cycles / sample and defined as 2*critical frequencies / fs). If fs is specified, Wn is in the same units as fs.
            - For analog filters, Wn is an angular frequency (e.g. rad/s).
        
        btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, default: lowpass
            - The type of filter. Default is 'lowpass'.
        
        output : {'ba', 'zpk', 'sos'}, default: 'ba'
            - Type of output: numerator/denominator ('ba'), pole-zero ('zpk'), or second-order sections ('sos'). Default is 'ba' for backwards compatibility, but 'sos' should be used for general-purpose filtering.
        
        fs : float, default: 1.0
            - The sampling frequency of the digital system.
        """
        coeffs= {}
        if implementation == 'ba':
            b, a = signal.butter(N,Wn,btype=btype,output=implementation,fs=fs)
            coeffs['a'] = a
            coeffs['b'] = b
        elif implementation == 'zpk':
            z,p,k = signal.butter(N,Wn,btype=btype,output=implementation,fs=fs)
            coeffs['z'] = z
            coeffs['p'] = p
            coeffs['k'] = k
        else:
            sos = signal.butter(N,Wn,btype=btype,output=implementation,fs=fs)
            coeffs['sos'] = sos
        
        f = cls(sess,'butter',btype,implementation,Wn,fs,coeffs)
        
        # add the filter to the session
        sess.add_misc_bcip_obj(f)
        
        return f
    
    @classmethod
    def create_cheby1(cls,sess,N,rp,Wn,btype='lowpass',\
                     implementation='ba',fs=1.0):
        """
        Factory method to create a Chebyshev Type-I BCIP filter object

        Parameters
        ----------
        sess: BCIPy Session Object
            - Session where the filter object will exist

        N : int
            - The order of the filter.

        rp : float
            - The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number.
        
        Wn : array_like
            - A scalar or length-2 sequence giving the critical frequencies. For Type I filters, this is the point in the transition band at which the gain first drops below -rp.
            - For digital filters, Wn are in the same units as fs. By default, fs is 2 half-cycles/sample, so these are normalized from 0 to 1, where 1 is the Nyquist frequency. (Wn is thus in half-cycles / sample.)
            - For analog filters, Wn is an angular frequency (e.g., rad/s).

        btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, default: lowpass
            - The type of filter. Default is 'lowpass'.
        
        output : {'ba', 'zpk', 'sos'}, default: 'ba'
            - Type of output: numerator/denominator ('ba'), pole-zero ('zpk'), or second-order sections ('sos'). Default is 'ba' for backwards compatibility, but 'sos' should be used for general-purpose filtering.
       
        fs : float, default: 1.0
            - The sampling frequency of the digital system.

        """
        coeffs= {}
        if implementation == 'ba':
            b, a = signal.cheby1(N,rp,Wn,btype=btype,output=implementation,fs=fs)
            coeffs['a'] = a
            coeffs['b'] = b
        elif implementation == 'zpk':
            z,p,k = signal.cheby1(N,rp,Wn,btype=btype,output=implementation,fs=fs)
            coeffs['z'] = z
            coeffs['p'] = p
            coeffs['k'] = k
        else:
            sos = signal.cheby1(N,rp,Wn,btype=btype,output=implementation,fs=fs)
            coeffs['sos'] = sos
        
        f = cls(sess,'cheby1',btype,implementation,Wn,fs,coeffs)
        
        # add the filter to the session
        sess.add_misc_bcip_obj(f)
        
        return f
    
    @classmethod
    def create_cheby2(cls,sess,N,rs,Wn,btype='lowpass',\
                     implementation='ba',fs=1.0):
        """
        Factory method to create a Chebyshev Type-II BCIPy filter object

        Parameters
        ----------
        sess: BCIPy Session Object
            - Session where the filter object will exist

        N : int
            - The order of the filter.

        rs : float
            - The minimum attenuation required in the stop band. Specified in decibels, as a positive number.
        
        Wn : array_like
            - A scalar or length-2 sequence giving the critical frequencies. For Type II filters, this is the point in the transition band at which the gain first reaches -rs.
            - For digital filters, Wn are in the same units as fs. By default, fs is 2 half-cycles/sample, so these are normalized from 0 to 1, where 1 is the Nyquist frequency. (Wn is thus in half-cycles / sample.)
            - For analog filters, Wn is an angular frequency (e.g., rad/s).
        
        btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, default: 'lowpass'
            - The type of filter. Default is 'lowpass'.
        
        output : {'ba', 'zpk', 'sos'}, default: 'ba'
            - Type of output: numerator/denominator ('ba'), pole-zero ('zpk'), or second-order sections ('sos'). Default is 'ba' for backwards compatibility, but 'sos' should be used for general-purpose filtering.
        
        fs : float, default: 1.0
            - The sampling frequency of the digital system.
        """
        coeffs= {}
        if implementation == 'ba':
            b, a = signal.cheby2(N,rs,Wn,btype=btype,output=implementation,fs=fs)
            coeffs['a'] = a
            coeffs['b'] = b
        elif implementation == 'zpk':
            z,p,k = signal.cheby2(N,rs,Wn,btype=btype,output=implementation,fs=fs)
            coeffs['z'] = z
            coeffs['p'] = p
            coeffs['k'] = k
        else:
            sos = signal.cheby2(N,rs,Wn,btype=btype,output=implementation,fs=fs)
            coeffs['sos'] = sos
        
        f = cls(sess,'cheby2',btype,implementation,Wn,fs,coeffs)
        
        # add the filter to the session
        sess.add_misc_bcip_obj(f)
        
        return f
        
    @classmethod
    def create_ellip(cls,sess,N,rp,rs,Wn,btype='lowpass',\
                     implementation='ba',fs=1.0):
        """
        Factory method to create a Elliptic BCIP filter object
        
        Parameters
        ----------
        N : int
            - The order of the filter.
        
        rp : float
            - The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number.
        
        rs : float
            - The minimum attenuation required in the stop band. Specified in decibels, as a positive number.
        
        Wn : array_like
            - A scalar or length-2 sequence giving the critical frequencies. For elliptic filters, this is the point in the transition band at which the gain first drops below -rp.
            - For digital filters, Wn are in the same units as fs. By default, fs is 2 half-cycles/sample, so these are normalized from 0 to 1, where 1 is the Nyquist frequency. (Wn is thus in half-cycles / sample.)
            - For analog filters, Wn is an angular frequency (e.g., rad/s).
        
        btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
            - The type of filter. Default is 'lowpass'.
        
        analog : bool, optional
            - When True, return an analog filter, otherwise a digital filter is returned.
        
        output : {'ba', 'zpk', 'sos'}, optional
            - Type of output: numerator/denominator ('ba'), pole-zero ('zpk'), or second-order sections ('sos'). Default is 'ba' for backwards compatibility, but 'sos' should be used for general-purpose filtering.
        
        fs : float, optional
            - The sampling frequency of the digital system.

        """
        coeffs= {}
        if implementation == 'ba':
            b, a = signal.ellip(N,rp,rs,Wn,\
                                btype=btype,output=implementation,fs=fs)
            coeffs['a'] = a
            coeffs['b'] = b
        elif implementation == 'zpk':
            z,p,k = signal.ellip(N,rp,rs,Wn,\
                                 btype=btype,output=implementation,fs=fs)
            coeffs['z'] = z
            coeffs['p'] = p
            coeffs['k'] = k
        else:
            sos = signal.ellip(N,rp,rs,Wn,\
                               btype=btype,output=implementation,fs=fs)
            coeffs['sos'] = sos
        
        f = cls(sess,'ellip',btype,implementation,Wn,fs,coeffs)
        
        # add the filter to the session
        sess.add_misc_bcip_obj(f)
        
        return f
    
    @classmethod
    def create_bessel(cls,sess,N,Wn,btype='lowpass',\
                     implementation='ba',norm='phase',fs=1.0):
        """
        Factory method to create a Bessel BCIP filter object

        Parameters
        ----------
        N : int
            - The order of the filter.
        
        Wn : array_like
            - A scalar or length-2 sequence giving the critical frequencies (defined by the norm parameter). For analog filters, Wn is an angular frequency (e.g., rad/s).
            - For digital filters, Wn are in the same units as fs. By default, fs is 2 half-cycles/sample, so these are normalized from 0 to 1, where 1 is the Nyquist frequency. (Wn is thus in half-cycles / sample.)
        
        btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
            - The type of filter. Default is 'lowpass'.
        
        analog : bool, optional
            - When True, return an analog filter, otherwise a digital filter is returned. (See Notes.)
        
        output : {'ba', 'zpk', 'sos'}, optional
            - Type of output: numerator/denominator ('ba'), pole-zero ('zpk'), or second-order sections ('sos'). Default is 'ba'.
        
        norm : {'phase', 'delay', 'mag'}, optional
            - Critical frequency normalization:
                - phase
                    - The filter is normalized such that the phase response reaches its midpoint at angular (e.g. rad/s) frequency Wn. This happens for both low-pass and high-pass filters, so this is the "phase-matched" case.
                    - The magnitude response asymptotes are the same as a Butterworth filter of the same order with a cutoff of Wn.
                    - This is the default, and matches MATLAB's implementation.

                - delay
                    - The filter is normalized such that the group delay in the passband is 1/Wn (e.g., seconds). This is the "natural" type obtained by solving Bessel polynomials.
                    
                - mag
                    - The filter is normalized such that the gain magnitude is -3 dB at angular frequency Wn.

        fs : float, optional
            - The sampling frequency of the digital system.
        """
        coeffs= {}
        if implementation == 'ba':
            b, a = signal.bessel(N,Wn,btype=btype,output=implementation,fs=fs)
            coeffs['a'] = a
            coeffs['b'] = b
        elif implementation == 'zpk':
            z,p,k = signal.bessel(N,Wn,btype=btype,output=implementation,fs=fs)
            coeffs['z'] = z
            coeffs['p'] = p
            coeffs['k'] = k
        else:
            sos = signal.bessel(N,Wn,btype=btype,output=implementation,fs=fs)
            coeffs['sos'] = sos
        
        f = cls(sess,'bessel',btype,implementation,Wn,fs,coeffs)
        
        # add the filter to the session
        sess.add_misc_bcip_obj(f)
        
        return f
