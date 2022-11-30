from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .utils.data_extraction import extract_nested_data

from classes.scalar import Scalar

import numpy as np

class ReducedSumKernel(Kernel):
    """
    Kernel to compute the sum of the input tensor's 
    element along the provided axis

    Parameters
    ----------
    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor object
        - Input trial data

    outA : Tensor or Scalar object
        - Output trial data

    axis : int or tuple of ints, default = None
        - What is this for

    keep_dims : bool, default = False
        - Or this

    """
    
    def __init__(self,graph,inA,outA,axis=None,keep_dims=False):
        super().__init__('ReducedSum',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA
        self._axis = axis
        self._keep_dims = keep_dims

        self._init_inA = None
        self._init_outA = None
    
        self._labels = None
    

    def _compute_output_sz(self, input_sz):
        if self._axis != None:
            axis = (self._axis,)
        else:
            axis = ()
        
        if self._keep_dims:
            # all reduced dimensions will be '1'
            out_shape = tuple([1 if i in axis else inA_shape[i] 
                                          for i in range(len(inA_shape))])
        elif axis == ():
            out_shape = (1,)
        else:
            out_shape = tuple([inA_shape[i] for i in range(len(inA_shape))
                                                   if i not in axis])
        
        return out_shape



    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """

        if sts == BcipEnums.SUCCESS and self._init_outA != None:
            # adjust the shape of init output tensor, as needed
            if len(self._init_outA.shape) == 0:
                input_sz = list(self._init_inA.shape)
                output_sz = self._compute_output_sz(input_sz)
                self._outA.shape = output_sz

            sts = self._process_data(self._init_inA, self._init_outA)

        return sts
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the inputs and outputs are the appropriate type
        if self._inA._bcip_type != BcipEnums.TENSOR:
            return BcipEnums.INVALID_PARAMETERS
        
        if (self._outA._bcip_type != BcipEnums.TENSOR and
            self._outA._bcip_type != BcipEnums.SCALAR):
            return BcipEnums.INVALID_PARAMETERS

        if (self._outA._bcip_type == BcipEnums.SCALAR and 
            (self._outA.data_type not in Scalar.valid_numeric_types())):
            return BcipEnums.INVALID_PARAMETERS
        
        inA_shape = self._inA.shape
        out_shape = self._compute_output_sz(inA_shape)

        # if the output is a virtual tensor and has no defined shape, set the shape now
        if (self._outA._bcip_type == BcipEnums.TENSOR 
            and self._outA.virtual 
            and len(self._outA.shape) == 0):
            self._outA.shape = out_shape
        
        # ensure the output shape equals the expected output shape
        if self._outA._bcip_type == BcipEnums.TENSOR and self._outA.shape != out_shape:
            return BcipEnums.INVALID_PARAMETERS
        elif self._outA._bcip_type == BcipEnums.SCALAR and out_shape != (1,):
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
        
        return self._process_data(self._inA, self._outA)
    
    @classmethod
    def add_reduced_sum_node(cls,graph,inA,outA,axis=None,keep_dims=False):
        """
        Factory method to create a reduced sum kernel 
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor object
            - Input trial data

        outA : Tensor or Scalar object
            - Output trial data

        axis : int or tuple of ints, default = None
            - What is this for

        keep_dims : bool, default = False
            - Or this

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
