# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:08:38 2019

Kernel.py - Defines a generic kernel class

@author: ivanovn
"""

from .bcip import BCIP
from .bcip_enums import BcipEnums
from abc import ABC, abstractmethod

class Kernel(BCIP, ABC):
    """
    An abstract base class that defines the minimum set of kernel methods that
    must be defined
    """
    
    def __init__(self,name,init_style,graph):
        session = graph.getSession()
        super().__init__(BcipEnums.KERNEL,session)
        self._name = name
        self._init_style = init_style
        
    # API Getters
    @property
    def name(self):
        return self._name
    
    @property
    def init_style(self):
        return self._init_style
    
    @abstractmethod
    def verify(self):
        pass
    
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def initialize(self):
        pass