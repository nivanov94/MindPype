"""
Created on Wed Mar 11 10:53:18 2020

@author: ivanovn
"""
#TODO: Delete

from ..classes.kernel import Kernel
from ..classes.node import Node
from ..classes.parameter import Parameter
from ..classes.tensor import Tensor
from ..classes.scalar import Scalar
from ..classes.bcip_enums import BcipEnums

import numpy as np
from scipy.stats import combine_pvalues

class CombinePValuesKernel(Kernel):
    """
    Combines p-values within a 1D tensor using Fisher's or Stouffer's methods
    """
    
    def __init__(self,graph,inA,out_ts,out_pv,method):
        """
        Kernel takes tensor input of RVs
        """
        super().__init__('CombinePValues',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._out_ts = out_ts
        self._out_pv = out_pv
        self._method = method    

        self._init_inA = None
        self._init_outA = None    
    
    def initialize(self):
        """
        No internal state to setup
        """
        return BcipEnums.SUCCESS
        
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        if (not isinstance(self._method,str) or 
            not (self._method.lower() in ('fisher','stouffer'))):
            return BcipEnums.INVALID_PARAMETERS
        
        
        
        # ensure the input is a tensor
        if not isinstance(self._inA,Tensor):
            return BcipEnums.INVALID_PARAMETERS
        
        input_shape = self._inA.shape
        if len(input_shape) < 1 or len(input_shape) > 2:
            return BcipEnums.INVALID_PARAMETERS
        
        
        if self._out_ts == None and self._out_pv == None:
            return BcipEnums.INVALID_PARAMETERS
        
        for output in (self._out_ts,self._out_pv):
            if (output != None and 
                (not isinstance(output,Tensor) and not isinstance(output,Scalar))):
                return BcipEnums.INVALID_PARAMETERS
            
            # if the output is a virtual tensor and dimensionless, 
            # add the dimensions now
            if isinstance(output,Tensor):
                output_shape = (1,1)
                if (output.virtual and len(output.shape) == 0):
                    output.shape = (1,1)
            
                if output.shape != output_shape:
                    return BcipEnums.INVALID_PARAMETERS
            
            elif isinstance(output,Scalar):
                # scalar output
                if output.data_type != float:
                    return BcipEnums.INVALID_PARAMETERS
  
        return BcipEnums.SUCCESS

    def initialization_execution(self):
        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data, output_data):
        try:
            (ts,pv) = combine_pvalues(np.squeeze(input_data.data),
                                      method=self._method)
            
            for result_value, output in zip((ts,pv),(self._out_ts,self._out_pv)):
                if isinstance(output,Tensor):
                    output.data = np.asarray(((result_value))) # make the result a 1x1 tensor
                elif isinstance(output,Scalar):
                    output.data = result_value
            
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel and calculate the combined p-value
        """
        
        try:
            (ts,pv) = combine_pvalues(np.squeeze(self._inA.data),
                                      method=self._method)
            
            for result_value, output in zip((ts,pv),(self._out_ts,self._out_pv)):
                if isinstance(output,Tensor):
                    output.data = np.asarray(((result_value))) # make the result a 1x1 tensor
                elif isinstance(output,Scalar):
                    output.data = result_value
            
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_combine_pvalues_node(cls,graph,inA,out_ts=None,out_pv=None,method='fisher'):
        """
        Factory method to create a normal CDF node
        """
        
        # create the kernel object
        k = cls(graph,inA,out_ts,out_pv,method)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),)
        
        if out_ts != None:
            params += (Parameter(out_ts,BcipEnums.OUTPUT),)
        
        if out_pv != None:
            params += (Parameter(out_pv,BcipEnums.OUTPUT),)
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
