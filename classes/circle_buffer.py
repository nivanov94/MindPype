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
    
    def __init__(self,sess,capacity):
        super().__init__(sess,capacity)
        
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
        
    @classmethod
    def create(cls,sess,capacity):
        cb = cls(sess,capacity)
        
        # add to the session
        sess.addData(cb)
        
        return cb