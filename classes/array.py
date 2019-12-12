# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:08:19 2019

Array.py - Defines class of array objects that contain other BCIP objects

@author: ivanovn
"""

from .bcip import BCIP
from .bcip_enums import BcipEnums

class Array(BCIP):
    """
    Array containing instances of other BCIP classes
    """
    
    def __init__(self,sess,capacity,virtual=False):
        super().__init__(BcipEnums.ARRAY,sess)
        
        self._num_items = 0 # keep track of the non-none elements
        self._capacity = capacity
        self._virtual = virtual
        
        # private non-user interfacing attributes
        self._elements = [None] * capacity
        
    
    def get_element(self,index):
        if index >= self.capacity or index < 0:
            return
        
        return self._elements[index]
    
    def set_element(self,index,element):
        if index >= self.capacity or index < 0:
            return False
        
        # if there was nothing there, increase the num items counter
        if self._element[index] == None:
            self._num_items += 1
        
        self._element[index] = element
        return True
    
    # User Facing Getters
    @property
    def capacity(self):
        return self._capacity
    
    @property
    def num_items(self):
        return self._num_items
    
    @property
    def virtual(self):
        return self._virtual
    
    @capacity.setter
    def capacity(self,capacity):
        if self.virtual:
            self._capacity = capacity
            self._elements = [None] * capacity
            
            
    def copy(self):
        """
        Create and return a deep copy of the array
        The copied array will maintain references to the same objects.
        If a copy of these is also desired, they will need to be copied
        separately.
        """
        cpy = Array(self.session,
                    self.capacity,
                    self.virtual)
        
        for e in range(self.num_items):
            cpy.set_element(e,self.get_element(e))
            
        return cpy
        
    
    # API constructor
    @classmethod
    def create(cls,sess,capacity):
        a = cls(sess,capacity)
        
        # add the array to the session
        sess.add_data(a)
        return a

    @classmethod
    def create_virtual(cls,sess,capacity=0):
        a = cls(sess,capacity,True)
        
        sess.add_data(a)
        
        return a