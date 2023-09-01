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

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """
        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # update output size, as needed
            if init_out.virtual:
                init_out.shape = init_in.shape[-2:] # TODO what are the expected inputs? will we ever compute more than one mean here?

            self._process_data(init_inputs, init_outputs)
            

    def _process_data(self, inputs, outputs):
        outputs[0].data = mean_riemann(inputs[0].data, sample_weight=self._w)

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
    
