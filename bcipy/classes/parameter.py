# -*- coding: utf-8 -*-
"""
Parameter.py - General Parameter class for BCIP

@author: ivanovn
"""

class Parameter:
    """
    Parameter class can be used to abstract data types as inputs and outputs 
    to nodes
    """
    
    def __init__(self,data,direction):
        self._data = data # reference of the data object represented by parameter
        self._direction = direction # enum indicating whether this is an input or output
    
    @property
    def direction(self):
        return self._direction
    
    @property
    def data(self):
        return self._data