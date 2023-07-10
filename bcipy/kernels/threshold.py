from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Scalar, Tensor

import numpy as np


class ThresholdKernel(Kernel):
    """
    Determine if scalar or tensor data elements are above or below threshold
    
    Parameters
    ----------

    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor or Scalar 
        Input trial data

    outA : Tensor or Scalar 
        Output trial data

    thresh : float
        Threshold value 

    """
    
    def __init__(self,graph,inA,outA,thresh):
        super().__init__('Threshold',BcipEnums.INIT_FROM_NONE,graph)
        self._in   = inA
        self._out  = outA
        self._thresh = thresh

        self._init_inA = None
        self._init_outA = None
        
        self._init_labels_in = None
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS

        if self._init_outA is not None and (self._init_inA is not None and self._init_inA.shape != ()):
            # set the output size, as needed
            if self._init_outA.virtual:
                self._init_outA.shape = self._init_inA.shape

            sts = self._process_data(self._init_inA, self._init_outA)

            # pass on the labels
            if self._init_labels_in._bcip_type != BcipEnums.TENSOR:
                input_labels = self._init_labels_in.to_tensor()
            else:
                input_labels = self._init_labels_in
            input_labels.copy_to(self._init_labels_out)
        
        return sts
        
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """

        # input/output must be a tensor or scalar
        if not ((self._inA._bcip_type == BcipEnums.TENSOR and self._outA._bcip_type == BcipEnums.TENSOR) or 
                (self._inA._bcip_type == BcipEnums.SCALAR and self._outA._bcip_type == BcipEnums.SCALAR)):
            return BcipEnums.INVALID_PARAMETERS

        if self._inA._bcip_type == BcipEnums.TENSOR:
            # input tensor must contain some values
            if len(self._inA.shape) == 0:
                return BcipEnums.INVALID_PARAMETERS

        if self._thresh._bcip_type != BcipEnums.SCALAR:
            return BcipEnums.INVALID_PARAMETERS

        if not self._thresh.data_type in Scalar.valid_numeric_types():
            return BcipEnums.INVALID_PARAMETERS


        if self._outA._bcip_type == BcipEnums.TENSOR:
            if self._outA.virtual and len(self._outA.shape) == 0:
                self._outA.shape = self._inA.shape

            if self._outA.shape != self._inA.shape:
                return BcipEnums.INVALID_PARAMETERS

        else:
            if not (self._inA.data_type in Scalar.valid_numeric_types()):
                return BcipEnums.INVALID_PARAMETERS

            if self._outA.data_type != self._inA.data_type:
                return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS


    def _process_data(self, input_data, output_data):
        """
        Process data according to outlined kernel method
        """
        try:
            if isinstance(input_data,Tensor):
                output_data.data = input_data.data > self._thresh.data
            else:
                gt = input_data.data > self._thresh.data
                if output_data.data_type == bool:
                    output_data.data = gt
                else:
                    output_data.data = int(gt)
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS
    

    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        return self._process_data(self._inA, self._inB, self._outA)
    
    @classmethod
    def add_threshold_node(cls,graph,inA,outA,thresh):
        """
        Factory method to create a threshold value kernel 
        and add it to a graph as a generic node object.

        Parameters
        ----------

        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor or Scalar 
            Input trial data

        outA : Tensor or Scalar 
            Output trial data

        thresh : float
            Threshold value
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,thresh)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT),
                  Parameter(thresh,BcipEnums.INPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
