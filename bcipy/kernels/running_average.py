# -*- coding: utf-8 -*-
"""
Created on Thurs Aug 8 14:13:13 2022

@author: aaronlio
"""

from types import NoneType
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.array import Array
from classes.circle_buffer import CircleBuffer
from classes.bcip_enums import BcipEnums
import numpy as np

class RunningAverageKernel(Kernel):
    """
    Kernel to calculate running average across multiple trials in a session. Trials are automatically included into the next running average
    calculation. 

    Parameters
    ----------
    graph : graph object
        - The graph where the RunningAverageKernel object should be added
    
    inA : Tensor object
        - Single Trial input data to the RunningAverageKernel; should be a 2D Tensor or Scalar object

    outA : Tensor/Scalar object
        - Output Tensor to store output of mean trial calculation; should be the same size of the input tensor or a scalar.

    running_average_cap : int
        - Indicates the maximum number of trials that the running average kernel will be used to compute. Used to preallocate tensor to store previous trial data

    axis : None or 0:
        - Axis by which to calculate running average. Currently only supports mean across trials when axis = 0 (ie. Average Tensor layer values), or single value mean, axis = None

    
    Examples
    --------
    """
    def __init__(self, graph, inA, outA, running_average_cap, axis = 0):
        super().__init__('RunningAverage',BcipEnums.INIT_FROM_NONE,graph)

        self._graph = graph
        self._inA = inA 
        
        self._running_average_cap = running_average_cap
        self._axis = axis
        self._init_inA = None
        self._init_outA = None     
        self._outA = outA

    def verify(self):
        if not (isinstance(self._inA,Tensor) or isinstance(self._inA,Scalar)):
            return BcipEnums.INVALID_PARAMETERS

        self._prev_data = CircleBuffer.create(self.session, self._running_average_cap, Tensor.create(self.session, self._inA.shape))

        #Check that expected numpy output dims are the same as the _outA tensor
        input_shape = self._inA.shape
        
        if self._axis == None:
            output_shape = (1,1)
        elif self._axis == 0:
            shape = [x for i,x in enumerate(input_shape) if i != self._axis]
            output_shape = tuple(shape * len(input_shape)) 

        else:
            return BcipEnums.INVALID_PARAMETERS
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (self._outA.virtual and len(self._outA.shape) == 0):
            self._outA.shape = output_shape
        
        # check output shape
        if self._outA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS
  

        print(input_shape, output_shape)

        return BcipEnums.SUCCESS


    def initialize(self):
        if self._init_outA.__class__ != NoneType:
            return self.initialization_execution()
        
        return BcipEnums.SUCCESS

    def initialization_execution(self):
        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data, output_data):
        
        try:
            if isinstance(input_data, Tensor):
                if self._axis == None:
                    output_data.data = np.mean(input_data.data)
                else:
                    output_data.data = np.mean(input_data.data,axis=0)

            return BcipEnums.SUCCESS

        except:
            return BcipEnums.EXE_FAILURE
     

    def execute(self):

        stacked_data = np.zeros((self._prev_data.num_elements + 1, self._inA.shape[0], self._inA.shape[1]))
        stacked_data[0,:,:] = self._inA.data

        for i in range(1, self._prev_data.num_elements+1):
            stacked_data[i, :, :] = self._prev_data.get_queued_element(i).data

        if len(stacked_data.shape) == 2:
            stacked_data = stacked_data[np.newaxis, :, :]
        
        stacked_tensor = Tensor.create_from_data(self.session, stacked_data.shape, stacked_data)

        self._prev_data.enqueue(self._inA)

        return self.process_data(stacked_tensor, self._outA)

        """if self._prev_data.num_elements == 0:
            sts = self._prev_data.enqueue(self._inA)
            if sts == BcipEnums.SUCCESS:
                return self.process_data(self._inA, self._outA, True)
            else:
                return BcipEnums.EXE_FAILURE

        elif self._prev_data.num_elements == 1:
            stacked_data = np.array(self._inA.data)
            stacked_data = np.stack((self._prev_data.get_queued_element(0).data, stacked_data))
            stacked_tensor = Tensor.create_from_data(self.session, stacked_data.shape, stacked_data)
            self._prev_data.enqueue(self._inA)
            return self.process_data(stacked_tensor, self._outA, False)
        
        else:
            stacked_data = np.array(self._inA.data)
            for i in range(self._prev_data.num_elements-1, -1, -1):
                stacked_data = np.dstack((self._prev_data.get_queued_element(i).data, stacked_data))
            
            #Dealing with common dimension change issue with dstack
            if stacked_data.shape[0] == stacked_data.shape[1]:
                stacked_data = np.moveaxis(stacked_data, (0,1,2),(2,1,0))

            stacked_tensor = Tensor.create_from_data(self.session, stacked_data.shape, stacked_data)

            self._prev_data.enqueue(self._inA)
            return self.process_data(stacked_tensor, self._outA)"""



            #except:
            #    return BcipEnums.EXE_FAILURE

            
            
            


    @classmethod
    def add_running_average_node(cls, graph, inA, outA, running_average_cap, axis):
        kernel = cls(graph, inA, outA, running_average_cap, axis)

        params = (Parameter(inA,BcipEnums.INPUT), \
                  Parameter(outA,BcipEnums.OUTPUT))

        node = Node(graph, kernel, params)

        graph.add_node(node)

        return node
