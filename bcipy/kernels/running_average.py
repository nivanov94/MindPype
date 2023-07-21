# -*coding: utf-8 -*-
"""
Created on Thurs Aug 8 14:13:13 2022

@author: aaronlio
"""

from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor, CircleBuffer
from ..core import BcipEnums
from .kernel_utils import extract_nested_data
import numpy as np

class RunningAverageKernel(Kernel):
    """
    Kernel to calculate running average across multiple trials in a session. Trials are automatically included into the next running average
    calculation. 

    Parameters
    ----------
    inA : Tensor or Scalar
        Single Trial input data to the RunningAverageKernel; should be a 2D Tensor or Scalar object

    outA : Tensor or Scalar
        Output Tensor to store output of mean trial calculation; should be the same size of the input tensor or a scalar.

    running_average_len : int
        Indicates the maximum number of trials that the running average kernel will be used to compute. Used to preallocate tensor to store previous trial data

    axis : None or 0:
        Axis by which to calculate running average. Currently only supports mean across trials when axis = 0 (ie. Average Tensor layer values), or single value mean, axis = None

    
    """
    def __init__(self, graph, inA, outA, running_average_len, axis = 0, flush_on_init = False):
        super().__init__('RunningAverage',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]

        self._running_average_len = running_average_len
        self._flush_on_init = flush_on_init
        self._axis = axis
        self._data_buff = None

    def verify(self):

        d_in = self.inputs[0]
        d_out = self.outputs[0]

        #Check that input is a tensor or scalar
        if d_in.bcip_type != BcipEnums.TENSOR and d_in.bcip_type != BcipEnums.SCALAR:
            return BcipEnums.INVALID_PARAMETERS

        self._data_buff = CircleBuffer.create(self.session, self._running_average_len,
                                              Tensor.create(self.session, d_in.shape))

        #Check that expected numpy output dims are the same as the output tensor
        input_shape = d_in.shape
        
        if self._axis == None:
            output_shape = (1,1)
        elif self._axis == 0:
            shape = [x for i,x in enumerate(input_shape) if i != self._axis]
            output_shape = tuple(shape * len(input_shape)) 

        else:
            return BcipEnums.INVALID_PARAMETERS

        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (d_out.virtual and len(d_out.shape) == 0):
            d_out.shape = output_shape
        
        # check output shape
        if d_out.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS
  
        return BcipEnums.SUCCESS


    def initialize(self):
        """
        This kernel has no internal state to be initialized. Call initialization_execution if downstream nodes are missing training data.
        """

        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]

        if self._flush_on_init:
            self._data_buff.flush()

        if init_in is not None:
            # check that the input is a tensor or array
            accepted_inputs = (BcipEnums.ARRAY,BcipEnums.CIRCLE_BUFFER)
            if init_in.bcip_type not in accepted_inputs:
                return BcipEnums.INITIALIZATION_FAILURE

            # extract the data fron the input and place it into the buffer
            for i in range(init_in.num_elements):
                data = extract_nested_data(init_in,i)
                self._data_buff.enqueue(data)

        if init_out is not None:
            if init_out.virtual:
                init_out.shape = self.outputs[0].shape

            # extract the data from the buffer
            X = extract_nested_data(self._data_buff)
            init_out.data = np.mean(X,axis=self._axis)
        
        return BcipEnums.SUCCESS


    def execute(self):
        """
        Add input data to data object storing previous trials and process the stacked data
        """

        try:
            # enqueue the input data
            self._data_buff.enqueue(self.inputs[0])

            # extract the data from the buffer
            X = extract_nested_data(self._data_buff)
            self.outputs[0].data = np.mean(X,axis=self._axis)
        except:
            return BcipEnums.EXE_FAILURE

        return BcipEnums.SUCCESS


    @classmethod
    def add_running_average_node(cls, graph, inA, outA, running_average_len, axis=0, flush_on_init=False):
        """
        Factory method to create running average node and add it to the specified graph

        Parameters
        ----------
        graph : Graph
            The graph where the node object should be added
        
        inA : Tensor or Scalar
            Single Trial input data to the RunningAverageKernel; should be a 2D Tensor or Scalar object

        outA : Tensor or Scalar
            Output Tensor to store output of mean trial calculation; should be the same size of the input tensor or a scalar.

        running_average_len : int
            Indicates the maximum number of trials that the running average kernel will be used to compute. Used to preallocate tensor to store previous trial data

        axis : None or 0:
            Axis by which to calculate running average. Currently only supports mean across trials when axis = 0 (ie. Average Tensor layer values), or single value mean, axis = None

        flush_on_init : bool
            If true, flushes the buffer on initialization.
    
    """
        kernel = cls(graph, inA, outA, running_average_len, axis, flush_on_init)

        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))

        node = Node(graph, kernel, params)

        graph.add_node(node)

        return node
