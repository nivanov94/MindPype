# -*- coding: utf-8 -*-
"""
edge.py - Defines the edge class for BCIP

@author: ivanovn
"""

class Edge:
    """
    Edge class used by BCIP block to schedule graphs. Each edge object
    represents a different BCIP data object and stores the nodes that produce
    and consume that data.
    """
    
    def __init__(self,data):
        self._data = data
        self._producers = []
        self._consumers = []
        
    
    @property
    def producers(self):
        return self._producers
    
    @property
    def consumers(self):
        return self._consumers
    
    @property
    def data(self):
        return self._data
    
    def add_producer(self,producing_node):
        self.producers.append(producing_node)
    
    def add_consumer(self,consuming_node):
        self.consumers.append(consuming_node)

