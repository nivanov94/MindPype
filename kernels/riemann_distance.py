from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.array import Array
from classes.bcip_enums import BcipEnums

import numpy as np

from pyriemann.utils.distance import distance_riemann

class RiemannDistanceKernel(Kernel):
    """
    Calculates the Riemann mean of covariances contained in a tensor
    """
    
    def __init__(self,graph,inA,inB,outA):
        """
        Kernel computes pairwise distances between 2D tensors
        """
        super().__init__('RiemannDistance',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._inB  = inB
        self._outA = outA
        
    
    def initialize(self):
        """
        No internal state to setup
        """
        return BcipEnums.SUCCESS
        
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors or Arrays of Tensors
        if (not isinstance(self._inA,Tensor)) and \
            (not isinstance(self._inA,Array)):
                return BcipEnums.INVALID_PARAMETERS
        
        if (not isinstance(self._inB,Tensor)) and \
            (not isinstance(self._inB,Array)):
                return BcipEnums.INVALID_PARAMETERS
        
        if (not isinstance(self._outA,Tensor)) and \
            (not isinstance(self._outA,Array)):
                return BcipEnums.INVALID_PARAMETERS
        
        num_mats = []
        mat_sz = None
        for param in (self._inA,self._inB):
            if isinstance(param,Tensor):
                # ensure it is 3D or 2D
                param_rank = len(param.shape)
                if not param_rank in (2,3):
                    return BcipEnums.INVALID_PARAMETERS
                
                if mat_sz == None:
                    mat_sz = param.shape[-2:]
                elif param.shape[-2:] != mat_sz:
                    return BcipEnums.INVALID_PARAMETERS
                
                if param_rank == 3:
                    num_mats.append(param.shape[0])
                else:
                    num_mats.append(1)
            else:
                #ensure it is an array of 2D tensors
                param_rank = param.capacity
                for i in range(param_rank):
                    e = param.get_element(i)
                    if not isinstance(e,Tensor) or len(e.shape) != 2:
                        return BcipEnums.INVALID_PARAMETERS
                    
                    if mat_sz == None:
                        mat_sz = e.shape
                    elif mat_sz != e.shape:
                        return BcipEnums.INVALID_PARAMETERS
                    
                num_mats.append(param_rank)
            
        
        num_combos = num_mats[0]*num_mats[1]
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (self._outA.virtual and len(self._outA.shape) == 0):
            self._outA.shape = (num_combos,1)
        
        
        if isinstance(self._outA,Tensor) and \
           self._outA.shape != (num_combos,1):
            return BcipEnums.INVALID_PARAMETERS
        elif isinstance(self._outA,Array):
            if self._outA.capacity != num_combos:
                return BcipEnums.INVALID_PARAMETERS
            
            for i in range(self._outA.capacity):
                e = self._outA.get_element(i)
                if not (isinstance(e,Tensor) or isinstance(e,Scalar)) or \
                (isinstance(e,Tensor) and e.shape != (1,1)) or \
                (isinstance(e,Scalar) and e.data_type != float):
                    return BcipEnums.INVALID_PARAMETERS
                
  
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel and calculate the mean
        """


        def get_obj_data_at_index(obj,index,rank):
            if isinstance(obj,Tensor):
                if rank == 1 and len(obj.shape) == 2:
                    return obj.data
                else:
                    return obj.data[index,:,:]
            else:
                return obj.get_element(index).data
            
        def set_obj_data_at_index(obj,index,data):
            if isinstance(obj,Tensor):
                tensor_data = obj.data # need to extract and edit numpy array b/c tensor currently does not allow sliced modifications
                tensor_data[index,0] = data
                obj.data = tensor_data
            else:
                e = obj.get_element(index)
                if isinstance(e,Tensor):
                    e.data = np.asarray([[data]])
                else:
                    e.data = data
        
        num_mats = []
        for in_param in (self._inA,self._inB):
            if isinstance(in_param,Tensor):
                if len(in_param.shape) == 3:
                    m = in_param.shape[0]
                else:
                    m = 1
            else:
                m = in_param.capacity
        
            num_mats.append(m)
        
        
        for i in range(num_mats[0]):
            # extract the ith element from inA
            x = get_obj_data_at_index(self._inA,i,num_mats[0])
            
            for j in range(num_mats[1]):
                # extract the jth element from inB
                y = get_obj_data_at_index(self._inB,j,num_mats[1])
                
                try:
                    set_obj_data_at_index(self._outA,num_mats[1]*i+j,
                                          distance_riemann(x,y))
                
                except:
                    return BcipEnums.FAILURE
                    
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_riemann_distance_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a Riemann mean calculating kernel
        """
        
        # create the kernel object
        k = cls(graph,inA,inB,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(inB,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

