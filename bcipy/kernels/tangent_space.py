from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

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
        super().__init__("TangentSpaceKernel", BcipEnums.INIT_FROM_DATA, graph)
        self.inputs = [inA]
        self.outputs = [outA]

        if initialization_data is not None:
            self.init_inputs = [initialization_data]

        self._sample_weight = sample_weight
        self._tsupdate = tsupdate
        
        
    def verify(self):
        """
        Verify inputs and outputs are appropriate shape and type
        """

        d_in = self.inputs[0]
        d_out = self.outputs[0]

        if (d_in.bcip_type != BcipEnums.TENSOR or
            d_out.bcip_type != BcipEnums.TENSOR):
            return BcipEnums.INVALID_PARAMETERS

        if len(d_in.shape) not in (2, 3):
            return BcipEnums.INVALID_PARAMETERS

        if d_in.shape[-1] != d_in.shape[-2]:
            return BcipEnums.INVALID_PARAMETERS
    

        # if output is virtual, set the output dims 
        Nc = d_in.shape[-1]
        ts_vect_len = Nc*(Nc+1)//2
        if d_out.virtual and len(d_out.shape) == 0:
            if len(d_in.shape) == 3:
                output_shape = (d_in.shape[0], ts_vect_len)
            else:
                output_shape = (1, ts_vect_len)

            d_out.shape = output_shape

        # verify the output dimensions
        if d_out.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS
                
        return BcipEnums.SUCCESS

    def initialize(self):
        """
        Initialize internal state of the kernel and update initialization data if downstream nodes are missing data
        """
        sts = BcipEnums.SUCCESS

        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]

        if init_in.bcip_type != BcipEnums.TENSOR:
            sts = BcipEnums.INITIALIZATION_FAILURE

        # fit the tangent space
        if sts == BcipEnums.SUCCESS:
            try:
                self._tangent_space = TangentSpace()
                # add regularization
                r = 0.001
                init_in.data = (1-r)*init_in.data + r*np.eye(init_in.shape[1])
                self._tangent_space = self._tangent_space.fit(init_in.data, 
                                                              sample_weight=self._sample_weight)
            except:
                sts = BcipEnums.INITIALIZATION_FAILURE
        
        # compute init output
        if sts == BcipEnums.SUCCESS and init_in is not None and init_out is not None:
            # set output shape
            Nt, Nc, _ = init_in.shape
            if init_out.virtual:
                init_out.shape = (Nt, Nc*(Nc+1)//2)
            
            sts = self._process_data(init_in, init_out)

            # pass on the labels
            self.copy_init_labels_to_output()
        
        return sts
    
    def execute(self):
        """
        Execute single trial processing
        """
        return self._process_data(self.inputs[0], self.outputs[0])

    def _process_data(self, inA, outA):
        """
        Process data according to outlined kernel function
        """
        
        if len(inA.shape) == 2:
            local_input_data = np.expand_dims(inA.data,0)
        else:
            local_input_data = inA.data
            
        try:
            # add regularization
            r = 0.001
            local_input_data = (1-r)*local_input_data + r*np.eye(local_input_data.shape[1])
            outA.data = self._tangent_space.transform(local_input_data)
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS


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
        
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))

        node = Node(graph, kernel, params)

        graph.add_node(node)

        return node

