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
    
    def __init__(self,sess,capacity,element_template):
        super().__init__(BcipEnums.ARRAY,sess)
        
        self._virtual = False # no virtual arrays for now
        self._volatile = False # no volatile arrays for now...
        
        self._capacity = capacity
        
        self._elements = [None] * capacity
        
        for i in range(capacity):
            self._elements[i] = element_template.make_copy()
        
    
    def get_element(self,index):
        if index >= self.capacity or index < 0:
            return
        
        return self._elements[index]
    
    def set_element(self,index,element):
        if index >= self.capacity or index < 0:
            return BcipEnums.FAILURE
        
        element.copy_to(self._elements[index])
        return BcipEnums.SUCCESS
    
    # User Facing Getters
    @property
    def capacity(self):
        return self._capacity
    
    @property
    def virtual(self):
        return self._virtual
    
    @property
    def volatile(self):
        return self._volatile
    
    @capacity.setter
    def capacity(self,capacity):
        if self.virtual:
            self._capacity = capacity
            self._elements = [None] * capacity
            
            
    def make_copy(self):
        """
        Create and return a deep copy of the array
        The copied array will maintain references to the same objects.
        If a copy of these is also desired, they will need to be copied
        separately.
        """
        cpy = Array(self.session,
                    self.capacity,
                    self.get_element(0))
        
        for e in range(self.capacity):
            cpy.set_element(e,self.get_element(e))
            
        # add the copy to the session
        self.session.add_data(cpy)
            
        return cpy
    
    def copy_to(self,dest_array):
        """
        Copy all the attributes of the array to another array. Note
        these will reference the same objects within the element list
        """
        dest_array.capacity = self.capacity
        for i in range(self.capacity):
            sts = dest_array.set_element(i,self.get_element(i))
            
            if sts != BcipEnums.SUCCESS:
                return sts
        
        return BcipEnums.SUCCESS
    
    # API constructor
    @classmethod
    def create(cls,sess,capacity,element_template):
        a = cls(sess,capacity,element_template)
        
        # add the array to the session
        sess.add_data(a)
        return a
