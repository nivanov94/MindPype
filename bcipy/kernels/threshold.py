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

    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS

        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # set the output size, as needed
            if init_out.virtual:
                init_out.shape = init_in.shape

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
        thresh = self.inputs[1]

        # input/output must be a tensor or scalar
        if not ((d_in.bcip_type == BcipEnums.TENSOR and d_out.bcip_type == BcipEnums.TENSOR) or 
                (d_in.bcip_type == BcipEnums.SCALAR and d_out.bcip_type == BcipEnums.SCALAR)):
            return BcipEnums.INVALID_PARAMETERS

        if d_in.bcip_type == BcipEnums.TENSOR:
            # input tensor must contain some values
            if len(d_in.shape) == 0:
                return BcipEnums.INVALID_PARAMETERS

        if thresh.bcip_type != BcipEnums.SCALAR:
            return BcipEnums.INVALID_PARAMETERS

        if not thresh.data_type in Scalar.valid_numeric_types():
            return BcipEnums.INVALID_PARAMETERS

        if d_out.bcip_type == BcipEnums.TENSOR:
            if d_out.virtual and len(d_out.shape) == 0:
                d_out.shape = d_in.shape

            if d_out.shape != d_in.shape:
                return BcipEnums.INVALID_PARAMETERS

        else:
            if not (d_in.data_type in Scalar.valid_numeric_types()):
                return BcipEnums.INVALID_PARAMETERS

            if d_out.data_type != d_in.data_type:
                return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS


    def _process_data(self, input_data, output_data):
        """
        Process data according to outlined kernel method
        """
        try:
            thresh = self.inputs[1]
            if isinstance(input_data,Tensor):
                output_data.data = input_data.data > thresh.data
            else:
                gt = input_data.data > thresh.data
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
        return self._process_data(self._inputs[0],self._outputs[0])
    
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
        
        Returns
        -------
        node : Node
            Node object that was added to the graph containing the kernel
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
