# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:51:07 2019

filter.py - Defines the filter Class for BCIP

@author: ivanovn
"""

from .bcip import BCIP
from .bcip_enums import BcipEnums
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
        sess.addMiscBcipObj(f)
        
        return f
    
    @classmethod
    def create_cheby1(cls,sess,N,rp,Wn,btype='lowpass',\
                     implementation='ba',fs=1.0):
        """
        Factory method to create a Chebyshev Type-I BCIP filter object
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
        sess.addMiscBcipObj(f)
        
        return f
    
    @classmethod
    def create_cheby2(cls,sess,N,rs,Wn,btype='lowpass',\
                     implementation='ba',fs=1.0):
        """
        Factory method to create a Chebyshev Type-I BCIP filter object
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
        sess.addMiscBcipObj(f)
        
        return f
        
    @classmethod
    def create_ellip(cls,sess,N,rp,rs,Wn,btype='lowpass',\
                     implementation='ba',fs=1.0):
        """
        Factory method to create a Chebyshev Type-I BCIP filter object
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
        sess.addMiscBcipObj(f)
        
        return f
    
    @classmethod
    def create_bessel(cls,sess,N,Wn,btype='lowpass',\
                     implementation='ba',norm='phase',fs=1.0):
        """
        Factory method to create a Chebyshev Type-I BCIP filter object
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
        sess.addMiscBcipObj(f)
        
        return f