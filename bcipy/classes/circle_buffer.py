# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:22:41 2019

circle_buffer.py - Defines a circular buffer class for BCIP objects

@author: ivanovn
"""

from array import Array
from bcip_enums import BcipEnums

class CircleBuffer(Array):
    """
    A circular buffer for BCIP objects
    """
    
    def __init__(self,sess,capacity,element_template):
        super().__init__(sess,capacity,element_template)
        
        self._head = None
        self._tail = None
    
    @property
    def num_elements(self):
        """
        return the number of elements currently in the buffer
        """
        if self.is_empty():
            return 0
        else:
            return ((self._tail - self._head) % self.capacity) + 1
    
    def is_empty(self):
        if self._head == None and self._tail == None:
            return True
        else:
            return False
        
    def is_full(self):
        if self._head == ((self._tail + 1) % self.capacity):
            return True
        else:
            return False
    
    
    def get_queued_element(self,index):
        """
        extract and return the element at the index relative to the current
        head
        """
        if index > self.num_elements:
            return None
        
        abs_index = (index + self._head) % self.capacity
        
        return self.get_element(abs_index)
    
    def peek(self):
        if self.is_empty():
            return None
        
        return super(CircleBuffer,self).get_element(self._head)
    
    def enqueue(self,obj):
        if self.is_empty():
            self._head = 0
            self._tail = -1
            
        elif self.is_full():
            self._head = (self._head + 1) % self.capacity
        
        self._tail = (self._tail + 1) % self.capacity
        return super(CircleBuffer,self).set_element(self._tail,obj)
        
    def enqueue_chunk(self,cb):
        """
        enqueue a number of elements from another circle buffer into this
        circle buffer
        """
        
        while not cb.is_empty():
            element = cb.dequeue()
            sts = self.enqueue(element)
            
            if sts != BcipEnums.SUCCESS:
                return sts
        
        return BcipEnums.SUCCESS
            
    
    def dequeue(self):
        if self.is_empty():
            return None
        
        ret = super(CircleBuffer,self).get_element(self._head)
        
        if self._head == self._tail:
            self._head = None
            self._tail = None
        else:
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
        
        # TODO this should be handled using API methods instead
            # copy the head and tail as well
        cpy._tail = self._tail
        cpy._head = self._head
        
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
        
        if isinstance(dest_array,CircleBuffer):
            # copy the head and tail as well
            dest_array._tail = self._tail
            dest_array._head = self._head
            
        return BcipEnums.SUCCESS
    
    def flush(self):
        """
        Empty the buffer of all elements

        Returns
        -------
        BCIP Status code

        """
        while not self.is_empty():
            self.dequeue()
        
        return BcipEnums.SUCCESS
        
    @classmethod
    def create(cls,sess,capacity,element_template):
        cb = cls(sess,capacity,element_template)
        
        # add to the session
        sess.add_data(cb)
        
        return cb