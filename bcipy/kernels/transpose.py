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

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """
        init_in = init_inputs[0]
        init_out = init_outputs[0]
        
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
            
            self._process_data(init_inputs, init_outputs)

    def _process_data(self, inputs, outputs):
        """
        Process data according to outlined kernel function
        """
        outputs[0].data = np.transpose(inputs[0].data,axes=self._axes)

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
