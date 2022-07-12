"""
Created on Wed Mar 11 10:38:01 2020

@author: ivanovn
"""

from ..classes.kernel import Kernel
from ..classes.node import Node
from ..classes.parameter import Parameter
from ..classes.tensor import Tensor
from ..classes.bcip_enums import BcipEnums

from scipy.stats import norm, chi2

class CDFKernel(Kernel):
    """
    Calculates the CDF for a distribution given a RV as input
    
    current support normal and chi2 distributions
    """
    
    def __init__(self,graph,inA,outA,dist,df,loc,scale):
        """
        Kernel takes tensor input of RVs
        """
        super().__init__('CDF',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA
        self._dist = dist
        self._loc = loc
        self._scale = scale
        self._df = df        
    
        self._init_inA = None
        self._init_outA = None

    def initialize(self):
        """
        No internal state to setup
        """
        return self.initialization_execution()
        
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors
        if (not isinstance(self._inA,Tensor)) or \
            (not isinstance(self._outA,Tensor)):
                return BcipEnums.INVALID_PARAMETERS
        
        input_shape = self._inA.shape        
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (self._outA.virtual and len(self._outA.shape) == 0):
            self._outA.shape = input_shape
        
        
        # check that the dimensions of the output match the dimensions of
        # input
        if self._inA.shape != self._outA.shape:
            return BcipEnums.INVALID_PARAMETERS
        
        # check that the distribution is supported
        if not self._dist in ('norm','chi2'):
            return BcipEnums.INVALID_NODE
        
        if self._dist == 'chi2' and self._df == None:
            return BcipEnums.INVALID_NODE
        
        return BcipEnums.SUCCESS
        
    def initialization_execution(self):
        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data, output_data):
        try:
            if self._dist == 'norm':
                output_data.data = norm.cdf(input_data.data,
                                           loc=self._loc,
                                           scale=self._scale)
            elif self._dist == 'chi2':
                output_data.data = chi2.cdf(input_data.data,
                                           self._df,
                                           loc=self._loc,
                                           scale=self._scale)
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel and calculate the CDF
        """
        return self.process_data(self._inA, self._outA)
    
    @classmethod
    def add_cdf_node(cls,graph,inA,outA,dist='norm',df=None,loc=0,scale=1):
        """
        Factory method to create a CDF node
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,dist,df,loc,scale)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT), \
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
