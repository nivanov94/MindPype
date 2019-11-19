# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:08:19 2019

Array.py - Defines class of array objects that contain other BCIP objects

@author: ivanovn
"""

from bcip import BCIP
from bcip_types import BcipEnums

class Array(BCIP):
    """
    Array containing instances of other BCIP classes
    """
    
    def __init__(self,capacity):
        super().__init__(BcipEnums.ARRAY)
        
        self.capacity = capacity
        self._elements = [None] * capacity
        self.num_items = 0 # keep track of the non-none elements
        
    
    def getElement(self,index):
        if index >= self.capacity or index < 0:
            return
        
        return self._elements[index]
    
    def setElement(self,index,element):
        if index >= self.capacity or index < 0:
            return False
        
        self._element[index] = element
        return True
    
        