from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

from scipy import signal

class ResampleKernel(Kernel):
    """
    Kernel to resample timeseries data
    
    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor or Array 
        Input trial data

    factor: float
        Resample factor

    outA : Tensor 
        Resampled timeseries data
        
    axis :
        The axis that is to be resampled
    """
    
    def __init__(self,graph,inA,factor,outA,axis = 1):
        super().__init__('Resample',BcipEnums.INIT_FROM_NONE,graph)
        self._in = inA
        self._out = outA
        self._factor = factor
        self._axis = axis

        self._init_inA = None
        self._init_outA = None

        self._init_labels_in = None
        self._init_labels_out = None
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS
        
        if self._init_outA is not None and (self._init_inA is not None and self._init_inA.shape != ()):
            if len(self._init_inA.shape) == 3 and self._axis == 1:
                axis = 2
            else:
                axis = self._axis
            # set the output size, as needed
            if self._init_outA.virtual:
                output_shape = list(self._init_inA.shape)
                output_shape[axis] = int(output_shape[axis] * self._factor)
                self._init_outA.shape = tuple(output_shape)
            
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
        
        # input and output must be a tensor 
        if (self._in._bcip_type != BcipEnums.TENSOR or
            self._out._bcip_type != BcipEnums.TENSOR):
            return BcipEnums.INVALID_PARAMETERS
        
        if self._axis >= len(self._in.shape) or self._axis < -len(self._in.shape):
            return BcipEnums.INVALID_PARAMETERS
        
        # if output is virtual, set the dimensions
        output_shape = list(self._in.shape)
        output_shape[self._axis] = int(output_shape[self._axis] * self._factor)
        output_shape = tuple(output_shape)
        if self._out._virtual and len(self._out.shape) == 0:
            self._out.shape = output_shape
      
        if self._out.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS
        

    def _process_data(self, input_data, output_data):
        """
        Process trial data according to the Numpy function
        """
        if len(input_data.shape) == 3 and self._axis == 1:
            axis = 2
        else:
            axis = self._axis
        
        try:
            output_data.data = signal.resample(input_data.data,
                                               output_data.shape[axis],
                                               axis=axis)
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function
        """
        
        return self._process_data(self._in, self._out)
    
    @classmethod
    def add_resample_node(cls,graph,inA,factor,outA,axis=1):
        """
        Factory method to create an extract kernel 
        and add it to a graph as a generic node object.

         graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor or Array 
            Input trial data

        factor: float
            Resample factor

        outA : Tensor 
            Resampled timeseries data
        
        axis :
            The axis that is to be resampled

        """
        
        # create the kernel object
        k = cls(graph,inA,factor,outA,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
