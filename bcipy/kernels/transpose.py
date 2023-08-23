from ..core import BcipEnums
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
        self.inputs = [inputA]
        self.outputs = [outputA]
        self._axes = axes

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

        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]
        
        if init_out is not None and (init_in is not None and init_in.shape != ()):
            
            if init_out.virtual:
                if self._axes == None:
                    init_axes = [_ for _ in range(len(init_in.shape))]
                    init_axes[-2:] = [init_axes[-1], init_axes[-2]]
                elif len(init_in.shape)+1 == len(self._axes):
                    init_axes = [0] + [a+1 for a in self._axes]
                else:
                    init_axes = self._axes
                    
                init_out.shape = self._compute_output_shape(init_in, init_axes)
            
            sts = self._process_data(init_in, init_out)

            # pass on the labels
            self.copy_init_labels_to_output()
        
        return sts

    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        d_in = self.inputs[0]
        d_out = self.outputs[0]

        # first ensure the input and output are tensors
        for param in (d_in, d_out):
            if param.bcip_type != BcipEnums.TENSOR:
                return BcipEnums.INVALID_PARAMETERS

        # check axes
        if (self._axes != None and
            len(self._axes) != len(d_in.shape)):
            return BcipEnums.INVALID_PARAMETERS
        
        # check the shape
        input_shape = d_in.shape
        input_rank = len(input_shape)
        
        if self._axes == None and input_rank != 2:
            return BcipEnums.INVALID_PARAMETERS

        # determine what the output shape should be
        output_shape = self._compute_output_shape(d_in, self._axes)
               
        # if the output is virtual and has no defined shape, set the shape now
        if d_out.virtual and len(d_out.shape) == 0:
            d_out.shape = output_shape
        
        # ensure the output tensor's shape equals the expected output shape
        if d_out.shape != output_shape:
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
        return self._process_data(self.inputs[0], self.outputs[0])


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
        
        Returns
        -------
        node : Node
            Node object that was added to the graph containing the kernel
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
