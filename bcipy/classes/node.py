# -*- coding: utf-8 -*-
"""
Node.py - Generic node class for BCIP

@author: ivanovn
"""

from .bcip import BCIP
from .bcip_enums import BcipEnums


class Node(BCIP):
    """
    Generic node object containing a kernel function

    Parameters
    ----------
    graph : Graph object
        - Graph where the Node object will exist
    kernel : Kernel Object
        - Kernel object to be used for processing within the Node
    params : dict
        - Dictionary of parameters outputted by kernel

    Attributes
    ----------
    _kernel : Kernel Object
        - Kernel object to be used for processing within the Node
    _params : dict
        - Dictionary of parameters outputted by kernel

    Examples
    --------
    >>> Node.create(example_graph, example_kernel, example_params)
    """
    
    def __init__(self,graph,kernel,params):
        sess = graph.session
        super().__init__(BcipEnums.NODE,sess)
        
        self._kernel = kernel
        self._params = params
        
    
    # API getters
    @property
    def kernel(self):
        return self._kernel
    
    def extract_inputs(self):
        """
        Return a list of all the node's inputs

        Parameters
        ----------
        None

        Return
        ------
        Array
            - List of inputs for the Node

        Examples
        --------

        >>> inputs = example_node.extract_inputs()
        >>> print(inputs)

            None

        """
        inputs = []
        for p in self._params:
            if p.direction == BcipEnums.INPUT:
                inputs.append(p.data)
        
        return inputs
    
    def extract_outputs(self):
        """
        Return a list of all the node's outputs

        Parameters
        ----------
        None

        Return
        ------
        Array
            - List of outputs for the Node

        Examples
        --------

        >>> inputs = example_node.extract_outputs()
        >>> print(inputs)

            None
        """
        outputs = []
        for p in self._params:
            if p.direction == BcipEnums.OUTPUT:
                outputs.append(p.data)
        
        return outputs
    
    def verify(self):
        """
        Verify the node is executable

        Parameters
        ----------
        None

        Return
        ------
        BCIP Status Code

        Examples
        --------
        >>> status = example_node.verify()
        >>> print(status)

            INVALID_PARAMETERS

        """
        return self.kernel.verify()
    
    def initialize(self):
        """
        Initialize the kernel function for execution
        
        Parameters
        ----------
        None

        Return
        ------
        BCIP Status Code

        Examples
        --------
        >>> status = example_node.initialize()
        >>> print(status)

            SUCCESS
        """
        return self.kernel.initialize()
      