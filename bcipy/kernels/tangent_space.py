"""
Created on Fri Aug 12 14:53:29 2022

tangent_space.py  

@author: aaronlio
"""

from pickle import NONE
from random import sample
from types import NoneType

from requests import session
from classes.kernel import Kernel
from classes.bcip_enums import BcipEnums
from classes.parameter import Parameter
from classes.node import Node
from classes.tensor import Tensor

from pyriemann.tangentspace import TangentSpace

class TangentSpaceKernel(Kernel):
    """
    Kernel to estimate Tangent Space
    

    Paramters
    ---------

    Examples
    --------
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
        if not isinstance(self._inA, Tensor):
            return BcipEnums.INVALID_PARAMETERS

        if len(self._inA.shape) not in (2, 3):
            return BcipEnums.INVALID_PARAMETERS

        if len(self._inA.shape) == 3:
            if self._inA.shape[1] != self._inA.shape[2]:
                return BcipEnums.INVALID_PARAMETERS
        else:
            if self._inA.shape[1] != self._inA.shape[0]:
                return BcipEnums.INVALID_PARAMETERS

        #change output dims
        self._outA.shape = (self._inA.shape[0], 2)
        return BcipEnums.SUCCESS

    def initialize(self):
        
        sts1, sts2 = BcipEnums.SUCCESS, BcipEnums.SUCCESS
        if self._initialization_data == None:
            self._initialization_data = self._init_inA
        
        #try:
        self._tangent_space = self._tangent_space.fit(self._initialization_data.data, None, sample_weight=self._sample_weight)
        #except:
        #    print("Tangent Space could not be properly fitted. Please check the shape of your initialization data")
        #    return BcipEnums.INITIALIZATION_FAILURE
        
        if self._init_outA.__class__ != NoneType:
            sts2 = self.initilization_execution()
        
        if sts1 != BcipEnums.SUCCESS:
            return sts1
        elif sts2 != BcipEnums.SUCCESS:
            return sts2
        else:
            return BcipEnums.SUCCESS
    
    def execute(self):
        return self.process_data(self._inA, self._outA)

    def process_data(self, input_data, output_data):
        try:  
            result = self._tangent_space.transform(input_data.data)
            if output_data.shape != result.shape:
                output_data.shape = result.shape
            output_data.data = result.data
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    def initilization_execution(self):
        sts = self.process_data(self._initialization_data, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    @classmethod
    def add_tangent_space_kernel(cls, graph, inA, outA, initialization_data, metric = 'riemann', tsupdate = False, sample_weight = None):

        kernel = cls(graph, inA, outA, initialization_data, metric, tsupdate, sample_weight)
        
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT)
                  )

        node = Node(graph, kernel, params)

        graph.add_node(node)

        return node

