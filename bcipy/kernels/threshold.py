from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Scalar, Tensor


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
        self.inputs = [inA, thresh]
        self.outputs = [outA]

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """
        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # set the output size, as needed
            if init_out.virtual:
                init_out.shape = init_in.shape

            self._process_data(init_inputs, init_outputs)

    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """

        d_in = self.inputs[0]
        d_out = self.outputs[0]
        thresh = self.inputs[1]

        # input/output must be a tensor or scalar
        if not ((d_in.bcip_type == BcipEnums.TENSOR and d_out.bcip_type == BcipEnums.TENSOR) or 
                (d_in.bcip_type == BcipEnums.SCALAR and d_out.bcip_type == BcipEnums.SCALAR)):
            raise TypeError("Threshold Kernel: Input and output must be either both tensors or both scalars")

        if d_in.bcip_type == BcipEnums.TENSOR:
            # input tensor must contain some values
            if len(d_in.shape) == 0:
                raise ValueError("Threshold Kernel: Input tensor must contain some values")

        if thresh.bcip_type != BcipEnums.SCALAR:
            raise TypeError("Threshold Kernel: Threshold value must be a scalar")

        if not thresh.data_type in Scalar.valid_numeric_types():
            raise TypeError("Threshold Kernel: Threshold value must be numeric")

        if d_out.bcip_type == BcipEnums.TENSOR:
            if d_out.virtual and len(d_out.shape) == 0:
                d_out.shape = d_in.shape

            if d_out.shape != d_in.shape:
                raise ValueError("Threshold Kernel: Input and output tensors must have the same shape")

        else:
            if not (d_in.data_type in Scalar.valid_numeric_types()):
                raise TypeError("Threshold Kernel: Input and output scalars must be numeric")

            if d_out.data_type != d_in.data_type:
                raise TypeError("Threshold Kernel: Input and output scalars must have the same data type")

    def _process_data(self, inputs, outputs):
        """
        Process data according to outlined kernel method
        """
        thresh = inputs[1]
        outputs[0].data = inputs[0].data > thresh.data
    
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
