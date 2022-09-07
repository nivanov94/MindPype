"""
Created on Fri Aug 12 14:53:29 2022

tangent_space.py  

@author: aaronlio
"""

from classes.kernel import Kernel
from classes.bcip_enums import BcipEnums
from classes.parameter import Parameter
from classes.node import Node
from classes.tensor import Tensor

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

        self._labels = None

        self._initialization_data = initialization_data

        self._init_inA = None
        self._init_outA = None
        self._sample_weight = sample_weight

        self._tangent_space = TangentSpace(metric, tsupdate)

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
                self._outA.shape = (self._inA.shape[0], ts_vect_len)
            else:
                self._outA.shape = (1, ts_vect_len)

        # verify the output dimensions
        if len(self._inA.shape) == 2:
            out_shape = (1,ts_vect_len)
        else:
            out_shape = (self._inA.shape[0], ts_vect_len)
            
        if self._outA.shape != out_shape:
            return BcipEnums.INVALID_PARAMETERS
            
                
        return BcipEnums.SUCCESS

    def initialize(self):
        """
        Initialize internal state of the kernel and update initialization data if downstream nodes are missing data
        """
        sts = BcipEnums.SUCCESS
        if self._initialization_data == None:
            self._initialization_data = self._init_inA
        
        try:
            self._tangent_space = self._tangent_space.fit(self._initialization_data.data, None, sample_weight=self._sample_weight)
        except:
            sts = BcipEnums.INITIALIZATION_FAILURE
        
        
        if sts == BcipEnums.SUCCESS and self._init_outA != None:
            Nt, Nc, _ = self._initialization_data.shape
            self._init_outA.shape = (Nt, Nc*(Nc+1)//2)
            sts = self._process_data(self._initialization_data, self._init_outA)
        
        return sts
    
    def execute(self):
        """
        Execute single trial processing
        """
        return self._process_data(self._inA, self._outA)

    def _process_data(self, input_data, output_data):
        """
        Process data according to outlined kernel function
        """
        
        if len(input_data.shape) == 2:
            local_input_data = np.expand_dims(input_data.data,0)
        else:
            local_input_data = input_data.data
            
        try:
            result = self._tangent_space.transform(local_input_data)
            output_data.data = result
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
                  Parameter(outA,BcipEnums.OUTPUT)
                  )

        node = Node(graph, kernel, params)

        graph.add_node(node)

        return node

