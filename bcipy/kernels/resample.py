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
        self.inputs = [inA]
        self.outputs = [outA]
        self._factor = factor
        self._axis = axis

    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS

        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]
        
        if init_out is not None and (init_in is not None and init_in.shape != ()):
            if len(init_in.shape) == 3 and self._axis == 1:
                axis = 2
            else:
                axis = self._axis
            # set the output size, as needed
            if init_out.virtual:
                output_shape = list(init_in.shape)
                output_shape[axis] = int(output_shape[axis] * self._factor)
                init_out.shape = tuple(output_shape)
            
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
        
        # input and output must be a tensor 
        if (d_in.bcip_type != BcipEnums.TENSOR or
            d_out.bcip_type != BcipEnums.TENSOR):
            return BcipEnums.INVALID_PARAMETERS
        
        if self._axis >= len(d_in.shape) or self._axis < -len(d_in.shape):
            return BcipEnums.INVALID_PARAMETERS
        
        # if output is virtual, set the dimensions
        output_shape = list(d_in.shape)
        output_shape[self._axis] = int(output_shape[self._axis] * self._factor)
        output_shape = tuple(output_shape)
        if d_out.virtual and len(d_out.shape) == 0:
            d_out.shape = output_shape
      
        if d_out.shape != output_shape:
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
        
        return self._process_data(self.inputs[0], self.outputs[0])
    
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
