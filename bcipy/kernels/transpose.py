from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

import numpy as np

class TransposeKernel(Kernel):
    """
    Kernel to compute the tensor transpose
    
    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inputA : Tensor or Scalar 
        Input trial data

    outputA : Tensor or Scalar 
        Output trial data

    axes : tuple or list of ints, optional
        If specified, it must be a tuple or list which contains a permutation of [0,1,..,N-1] where N is the number of axes of a. The i'th axis of the returned array will correspond to the axis numbered axes[i] of the input. If not specified, defaults to range(a.ndim)[::-1], which reverses the order of the axes.
    
    """
    
    def __init__(self,graph,inputA,outputA,axes):
        super().__init__('Transpose',BcipEnums.INIT_FROM_NONE,graph)
        self._inputA  = inputA
        self._outputA = outputA
        self._axes = axes

        self._init_inA = None
        self._init_outA = None

        self._init_labels_in = None
        self._init_labels_out = None
    

    def _compute_output_shape(self, inA, axes):
        # check the shape
        input_shape = inA.shape
        input_rank = len(input_shape)
        
        # determine what the output shape should be
        if input_rank == 0:
            return ()

        if axes == None:
            output_shape = reversed(input_shape)
        else:
            output_shape = input_shape[self._axes]

        return tuple(output_shape)

    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS
        
        if self._init_outA is not None and (self._init_inA is not None and self._init_inA.shape != ()):
            
            if self._init_outA.virtual:
                if self._axes == None:
                    init_axes = [_ for _ in range(len(self._init_inA.shape))]
                    init_axes[-2:] = [init_axes[-1], init_axes[-2]]
                elif len(self._init_inA)+1 == len(self._axes):
                    init_axes = [0] + [a+1 for a in self._axes]
                else:
                    init_axes = self._axes
                    
                self._init_outA.shape = self._compute_output_shape(self._init_inA, init_axes)
            
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
        
        # first ensure the input and output are tensors
        for param in (self._inA, self._outA):
            if param._bcip_type != BcipEnums.TENSOR:
                return BcipEnums.INVALID_PARAMETERS

        # check axes
        if (self._axes != None and
            len(self._axes) != len(self._inputA.shape)):
            return BcipEnums.INVALID_PARAMETERS
        
        # check the shape
        input_shape = self._inputA.shape
        input_rank = len(input_shape)
        
        if self._axes == None and input_rank != 2:
            return BcipEnums.INVALID_PARAMETERS

        # determine what the output shape should be
        output_shape = self._compute_output_shape(self._inA, self._axes)
               
        # if the output is virtual and has no defined shape, set the shape now
        if self._outputA.virtual and len(self._outputA.shape) == 0:
            self._outputA.shape = output_shape
        
        # ensure the output tensor's shape equals the expected output shape
        if self._outputA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS

    def _process_data(self, input_data, output_data):
        """
        Process data according to outlined kernel function
        """
        output_data.data = np.transpose(input_data.data,axes=self._axes)

        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using the numpy transpose function
        """
        return self._process_data(self._inputA, self._outputA)


    @classmethod
    def add_transpose_node(cls,graph,inputA,outputA,axes=None):
        """
        Factory method to create a transpose kernel and add it to a graph
        as a generic node object.

        Parameters
        ----------
        graph : Graph 
            Graph that the kernel should be added to

        inputA : Tensor or Scalar 
            Input trial data

        outputA : Tensor or Scalar 
            Output trial data

        axes : tuple or list of ints, default = None
            If specified, it must be a tuple or list which contains a permutation of [0,1,..,N-1] where N is the number of axes of a. The i'th axis of the returned array will correspond to the axis numbered axes[i] of the input. If not specified, defaults to range(a.ndim)[::-1], which reverses the order of the axes.
        

        """
        
        # create the kernel object
        k = cls(graph,inputA,outputA,axes)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT),
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
