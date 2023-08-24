from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

from pyriemann.utils.mean import mean_riemann

class RiemannMeanKernel(Kernel):
    """
    Calculates the Riemann mean of covariances contained in a tensor

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor 
        Input data

    outA : Tensor 
        Output trial data

    axis : int
        Axis over which the mean should be calculated (see np.mean for more info)

    weights : array_like
        Weights for each sample
    """
    
    def __init__(self,graph,inA,outA,weights):
        """
        Kernel takes 3D Tensor input and produces 2D Tensor representing mean
        """
        super().__init__('RiemannMean',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]

        self._w = weights

    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS

        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # update output size, as needed
            if init_out.virtual:
                init_out.shape = init_in.shape[-2:] # TODO what are the expected inputs? will we ever compute more than one mean here?

            sts = self._process_data(init_in, init_out)
            
            # pass on the labels - TODO would there be a reduction in dimensionality resulting in reduction in labels?
            self.copy_init_labels_to_output()
        
        return BcipEnums.SUCCESS
        

    def _process_data(self, input_data, output_data):
        output_data.data = mean_riemann(input_data.data,sample_weight=self._w)
        return BcipEnums.SUCCESS

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        d_in = self.inputs[0]
        d_out = self.outputs[0]

        # first ensure the input and output are tensors
        if (d_in.bcip_type != BcipEnums.TENSOR or 
            d_out.bcip_type != BcipEnums.TENSOR):
            return BcipEnums.INVALID_PARAMETERS
        
        input_shape = d_in.shape
        input_rank = len(input_shape)
        
        # input tensor must be rank 3
        if input_rank != 3:
            return BcipEnums.INVALID_PARAMETERS
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (d_out.virtual and len(d_out.shape) == 0):
            d_out.shape = input_shape[1:]
        
        
        # output tensor should be one dimensional
        if len(d_out.shape) > 2:
            return BcipEnums.INVALID_PARAMETERS
        
        # check that the dimensions of the output match the dimensions of
        # input
        if d_in.shape[1:] != d_out.shape:
            return BcipEnums.INVALID_PARAMETERS
        
        if self._w != None and len(self._w) != d_in.shape[0]:
            return BcipEnums.INVALID_PARAMETERS
  
        return BcipEnums.SUCCESS

        
    def execute(self):
        """
        Execute the kernel and calculate the mean
        """
        return self._process_data(self.inputs[0], self.outputs[0])
    
    @classmethod
    def add_riemann_mean_node(cls,graph,inA,outA,weights=None):
        """
        Factory method to create a Riemann mean calculating kernel

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor
            Input data
        
        outA : Tensor
            Output trial data

        weights : array_like, default=None
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,weights)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
