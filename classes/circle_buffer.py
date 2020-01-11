# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:22:41 2019

circle_buffer.py - Defines a circular buffer class for BCIP objects

@author: ivanovn
"""

from .array import Array

class CircleBuffer(Array):
    """
    A circular buffer for BCIP objects
    """
    
    def __init__(self,sess,capacity,element_template):
        super().__init__(sess,capacity,element_template)
        
        self._head = 0
        self._tail = -1
        
    def enqueue(self,obj):
        self._tail = (self._tail + 1) % self.capacity
        return super(CircleBuffer,self).setElement(self._tail,obj)
    
    def dequeue(self):
        # TODO check if its empty?
        ret = super(CircleBuffer,self).getElement(self._head)
        self._head = (self._head + 1) % self.capacity
        return ret
    
    def make_copy(self):
        """
        Create and return a deep copy of the array
        The copied array will maintain references to the same objects.
        If a copy of these is also desired, they will need to be copied
        separately.
        """
        cpy = CircleBuffer(self.session,
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
            dest_array.set_element(i,self.get_element(i))
        
    @classmethod
    def create(cls,sess,capacity,element_template):
        cb = cls(sess,capacity,element_template)
        
        # add to the session
        sess.add_data(cb)
        
        return cb