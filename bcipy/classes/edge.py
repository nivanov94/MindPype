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

    Parameters
    ----------
    data : Data object
        - The data to be stored within the Edge object

    Attributes
    ----------
    _producers : array of Node objects
        Node objects that will produce the data within the Edge object

    _consumers : array of Node objects
        Node objects that will consume the data within the Edge object

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

    def add_producer(self, producing_node):
        """
        Add a specified node as a producer to an Edge object

        Parameters
        ----------
        producing_node : Node object
            Node to be added as a producer to the referenced Edge object
        
        Examples
        --------
        example_edge.add_producer(example_producing_edge)

        Return
        ------
        None

        """
        self.producers.append(producing_node)

    def add_consumer(self, consuming_node):
        """
        Add a specified node as a consumer to an Edge object

        Parameters
        ----------
        consuming_node : Node object
            Node to be added as a consumer to the referenced Edge object
        
        Examples
        --------
        example_edge.add_consumer(example_consumer_edge)

        Return
        ------
        None
        """
        self.consumers.append(consuming_node)

    def add_data(self, data):
        """
        Add specified data to an Edge object

        Parameters
        ----------
        data : Tensor, Scalar, Array, Python Built-in Data Types
            Data to be added to the referenced Edge object
        
        Examples
        --------
        example_edge.add_data(example_data)

        Return
        ------
        None
        """
        self.data = data

