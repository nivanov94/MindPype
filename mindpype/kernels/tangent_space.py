from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor

from pyriemann.tangentspace import TangentSpace
import numpy as np

class TangentSpaceKernel(Kernel):
    """
    Kernel to estimate Tangent Space. Applies Pyriemann.tangentspace method

    Parameters
    ----------
    graph : Graph
        Graph object that this node belongs to
    inA : Tensor
        Input data
    outA : Tensor
        Output data
    initialization_data : Tensor
        Data to initialize the estimator with (n_trials, n_channels, n_samples)
    metric : str, default = 'riemann'
        See pyriemann.tangentspace for more info
    metric : bool, default = False
        See pyriemann.tangentspace for more info
    sample_weight : ndarray, or None, default = None
        sample of each weight. If none, all samples have equal weight

    """


    def __init__(self, graph, inA, outA, initialization_data = None, 
                 metric = 'riemann', tsupdate = False, sample_weight = None):
        super().__init__("TangentSpaceKernel", MPEnums.INIT_FROM_DATA, graph)
        self.inputs = [inA]
        self.outputs = [outA]

        if initialization_data is not None:
            self.init_inputs = [initialization_data]

        self._sample_weight = sample_weight
        self._tsupdate = tsupdate
        self._covariance_inputs = (0,)
        
        
    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Initialize internal state of the kernel and update initialization data if downstream nodes are missing data
        """
        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_in.mp_type != MPEnums.TENSOR:
            raise TypeError("Tangent Space Kernel: Initialization input must be a Tensor")

        # fit the tangent space
        self._tangent_space = TangentSpace()
        # add regularization
        r = 0.001 # TODO make this a parameter
        init_in.data = (1-r)*init_in.data + r*np.eye(init_in.shape[1])
        self._tangent_space = self._tangent_space.fit(init_in.data, 
                                                      sample_weight=self._sample_weight)
        
        # compute init output
        if init_in is not None and init_out is not None:
            # set output shape
            Nt, Nc, _ = init_in.shape
            if init_out.virtual:
                init_out.shape = (Nt, Nc*(Nc+1)//2)
            
            self._process_data(init_inputs, init_outputs)

    def _process_data(self, inputs, outputs):
        """
        Process data according to outlined kernel function
        """
        inA = inputs[0]

        if len(inA.shape) == 2:
            local_input_data = np.expand_dims(inA.data,0)
        else:
            local_input_data = inA.data
            
        # add regularization TODO make this a parameter
        r = 0.001
        local_input_data = (1-r)*local_input_data + r*np.eye(local_input_data.shape[1])
        outputs[0].data = self._tangent_space.transform(local_input_data)


    @classmethod
    def add_tangent_space_node(cls, graph, inA, outA, initialization_data = None, metric = 'riemann', tsupdate = False, sample_weight = None):
        """
        Factory method to create a tangent_space_kernel, add it to a node, and add the node to a specified graph

        Parameters
        ----------
        graph : Graph
            Graph object that this node belongs to
        inA : Tensor 
            Input data
        outA : Tensor 
            Output data
        initialization_data : Tensor 
            Data to initialize the estimator with (n_trials, n_channels, n_samples)
        metric : str, default = 'riemann'
            See pyriemann.tangentspace for more info
        metric : bool, default = False
            See pyriemann.tangentspace for more info
        sample_weight : ndarray, or None, default = None
            sample of each weight. If none, all samples have equal weight

        Returns
        -------
        node : Node
            Node object that was added to the graph
        """

        kernel = cls(graph, inA, outA, initialization_data, metric, tsupdate, sample_weight)
        
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))

        node = Node(graph, kernel, params)

        graph.add_node(node)

        return node
