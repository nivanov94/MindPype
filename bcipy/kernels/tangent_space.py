from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .utils.data_extraction import extract_nested_data

from pyriemann.tangentspace import TangentSpace
import numpy as np

class TangentSpaceKernel(Kernel):
    """
    Kernel to estimate Tangent Space. Applies Pyriemann.tangentspace method

    Paramters
    ---------
    inA : Tensor object
        - Input data

    outA : Tensor object
        - Output data

    initialization_data : Tensor object
        - Data to initialize the estimator with (n_trials, n_channels, n_samples)

    metric : str, default = 'riemann'
        - See pyriemann.tangentspace for more info

    metric : bool, default = False
        - See pyriemann.tangentspace for more info

    sample_weight : ndarray, or None, default = None
        - sample of each weight. If none, all samples have equal weight

    """


    def __init__(self, graph, inA, outA, initialization_data, metric = 'riemann', tsupdate = False, sample_weight = None):
        super().__init__("TangentSpaceKernel", BcipEnums.INIT_FROM_DATA, graph)
        self._inA = inA
        self._outA = outA

        self._init_inA = initialization_data
        self._init_outA = None
        self._init_labels_in = None
        self._init_labels_out = None
        
        
    def verify(self):
        """
        Verify inputs and outputs are appropriate shape and type
        """
        if (self._inA._bcip_type != BcipEnums.TENSOR or
            self._outA._bcip_type != BcipEnums.TENSOR):
            return BcipEnums.INVALID_PARAMETERS

        if len(self._inA.shape) not in (2, 3):
            return BcipEnums.INVALID_PARAMETERS

        if self._inA.shape[-1] != self._inA.shape[-2]:
            return BcipEnums.INVALID_PARAMETERS
    

        # if output is virtual, set the output dims 
        Nc = self._inA.shape[-1]
        ts_vect_len = Nc*(Nc+1)//2
        if self._outA._virtual and len(self._outA.shape) == 0:
            if len(self._inA.shape) == 3:
                output_shape = (self._inA.shape[0], ts_vect_len)
            else:
                output_shape = (1, ts_vect_len)

            self._outA.shape = output_shape

        # verify the output dimensions
        if self._outA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS
            
                
        return BcipEnums.SUCCESS

    def initialize(self):
        """
        Initialize internal state of the kernel and update initialization data if downstream nodes are missing data
        """
        sts = BcipEnums.SUCCESS
        if self._init_inA._bcip_type != BcipEnums.TENSOR:
            sts = BcipEnums.INITIALIZATION_FAILURE

        # fit the tangent space
        if sts == BcipEnums.SUCCESS:
            try:
                self._tangent_space = self._tangent_space.fit(self._init_inA.data, 
                                                              sample_weight=self._sample_weight)
            except:
                sts = BcipEnums.INITIALIZATION_FAILURE
        
        # compute init output
        if sts == BcipEnums.SUCCESS and self._init_outA != None:
            # set output shape
            Nt, Nc, _ = self._init_inA.shape
            self._init_outA.shape = (Nt, Nc*(Nc+1)//2)
            sts = self._process_data(self._init_inA, self._init_outA)

            # pass on the labels
            if self._init_labels_in._bcip_type != BcipEnums.TENSOR:
                input_labels = self._init_labels_in.to_tensor()
            else:
                input_labels = self._init_labels_in
            input_labels.copy_to(self._init_labels_out)
        
        return sts
    
    def execute(self):
        """
        Execute single trial processing
        """
        return self._process_data(self._inA, self._outA)

    def _process_data(self, inA, output_data):
        """
        Process data according to outlined kernel function
        """
        
        if len(inA.shape) == 2:
            local_input_data = np.expand_dims(inA.data,0)
        else:
            local_input_data = inA.data
            
        try:
            outA.data = self._tangent_space.transform(local_input_data)
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS


    @classmethod
    def add_tangent_space_node(cls, graph, inA, outA, initialization_data = None, metric = 'riemann', tsupdate = False, sample_weight = None):
        """
        Factory method to create a tangent_space_kernel, add it to a node, and add the node to a specified graph

        Paramters
        ---------
        inA : Tensor object
            - Input data

        outA : Tensor object
            - Output data

        initialization_data : Tensor object
            - Data to initialize the estimator with (n_trials, n_channels, n_samples)

        metric : str, default = 'riemann'
            - See pyriemann.tangentspace for more info

        metric : bool, default = False
            - See pyriemann.tangentspace for more info

        sample_weight : ndarray, or None, default = None
            - sample of each weight. If none, all samples have equal weight
        """

        kernel = cls(graph, inA, outA, initialization_data, metric, tsupdate, sample_weight)
        
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))

        node = Node(graph, kernel, params)

        graph.add_node(node)

        return node

