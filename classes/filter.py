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
        super().__init__(BcipEnums.FILTER)
        self.sess = sess
        
        self.ftype = ftype
        self.btype = btype
        self.implementation = implementation
        self.fs = fs
        self.crit_frqs = crit_frqs
        
        self.coeffs = coeffs
        
    
    @classmethod
    def createButter(cls,sess,N,Wn,btype='lowpass',implementation='ba',fs=1.0):
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
    def createCheby1(cls,sess,N,rp,Wn,btype='lowpass',\
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
    def createCheby2(cls,sess,N,rs,Wn,btype='lowpass',\
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
    def createEllip(cls,sess,N,rp,rs,Wn,btype='lowpass',\
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
    def createBessel(cls,sess,N,Wn,btype='lowpass',\
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