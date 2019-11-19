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
        self.data = data
        self.producer = []
        self.consumers = []
        
    
    def getProducers(self):
        return self.producer
    
    def getConsumers(self):
        return self.consumers
    
    def addProducer(self,producing_node):
        self.producer.append(producing_node)
    
    def addConsumer(self,consuming_node):
        self.consumers.append(consuming_node)

