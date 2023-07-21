from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .kernel_utils import extract_nested_data

from ..containers import Scalar

import numpy as np

class ReducedSumKernel(Kernel):
    """
    Kernel to compute the sum of the input tensor's 
    element along the provided axis

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor 
        Input trial data

    outA : Tensor or Scalar 
        Output trial data

    axis : int or tuple of ints, default = None
        What is this for

    keep_dims : bool, default = False
        Or this

    """
    
    def __init__(self,graph,inA,outA,axis=None,keep_dims=False):
        super().__init__('ReducedSum',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._axis = axis
        self._keep_dims = keep_dims


    def _compute_output_sz(self, input_sz):
        if self._axis != None:
            axis = (self._axis,)
        else:
            axis = ()
        
        if self._keep_dims:
            # all reduced dimensions will be '1'
            out_shape = tuple([1 if i in axis else input_sz[i] 
                                          for i in range(len(input_sz))])
        elif axis == ():
            out_shape = (1,)
        else:
            out_shape = tuple([input_sz[i] for i in range(len(input_sz))
                                                   if i not in axis])
        
        return out_shape



    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """

        sts = BcipEnums.SUCCESS

        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # adjust the shape of init output tensor, as needed
            if init_out.virtual:
                input_sz = list(init_in.shape)
                output_sz = self._compute_output_sz(input_sz)
                init_out.shape = output_sz

            sts = self._process_data(init_in, init_out)

            # pass on the labels
            self.copy_init_labels_to_output()

        return sts
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """

        inA = self.inputs[0]
        outA = self.outputs[0]
        
        # first ensure the inputs and outputs are the appropriate type
        if inA.bcip_type != BcipEnums.TENSOR:
            return BcipEnums.INVALID_PARAMETERS
        
        if (outA.bcip_type != BcipEnums.TENSOR and
            outA.bcip_type != BcipEnums.SCALAR):
            return BcipEnums.INVALID_PARAMETERS

        if (outA.bcip_type == BcipEnums.SCALAR and 
            (outA.data_type not in Scalar.valid_numeric_types())):
            return BcipEnums.INVALID_PARAMETERS
        
        inA_shape = inA.shape
        out_shape = self._compute_output_sz(inA_shape)

        # if the output is a virtual tensor and has no defined shape, set the shape now
        if (outA.bcip_type == BcipEnums.TENSOR 
            and outA.virtual 
            and len(outA.shape) == 0):
            outA.shape = out_shape
        
        # ensure the output shape equals the expected output shape
        if outA.bcip_type == BcipEnums.TENSOR and outA.shape != out_shape:
            return BcipEnums.INVALID_PARAMETERS
        elif outA.bcip_type == BcipEnums.SCALAR and out_shape != (1,):
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS

    def _process_data(self, input_data, output_data):
        try:
            output_data.data = np.sum(input_data.data, 
                                      axis=self._axis,
                                      keepdims=self._keep_dims)

        except:
            return BcipEnums.EXE_FAILURE
            
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        
        return self._process_data(self.inputs[0], self.outputs[0])
    
    @classmethod
    def add_reduced_sum_node(cls,graph,inA,outA,axis=None,keep_dims=False):
        """
        Factory method to create a reduced sum kernel 
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor 
            Input trial data

        outA : Tensor or Scalar 
            Output trial data

        axis : int or tuple of ints, default = None
            What is this for

        keep_dims : bool, default = False
            Or this

        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axis,keep_dims)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
