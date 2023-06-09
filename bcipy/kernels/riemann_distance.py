from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor

import numpy as np

from pyriemann.utils.distance import distance_riemann

class RiemannDistanceKernel(Kernel):
    """
    Calculates the Riemann mean of covariances contained in a tensor. Kernel computes pairwise distances between 2D tensors

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor or Array 
        First input data

    inB : Tensor or Array 
        Second Input data

    outA : Tensor or Scalar 
        Output trial data

    """
    
    def __init__(self,graph,inA,inB,outA):
        """
        Kernel computes pairwise distances between 2D tensors
        """
        super().__init__('RiemannDistance',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._inB  = inB
        self._outA = outA

        self._init_inA = None
        self._init_inB = None
        self._init_outA = None

        self._init_labels_in = None
        self._init_labels_out = None
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized. Call initialization_execution if downstream nodes are missing training data
        
        """

        sts = BcipEnums.SUCCESS

        if self._init_outA != None:
            # update output size, as needed
            if self._init_outA.virtual:
                output_sz = self._compute_output_shape(self._init_inA, self._init_inB)
                self._init_outA.shape = output_sz

            sts = self._process_data(self._init_inA, self._init_inB, self._init_outA)

            # pass on the labels
            if self._init_labels_in._bcip_type != BcipEnums.TENSOR:
                input_labels = self._init_labels_in.to_tensor()
            else:
                input_labels = self._init_labels_in
            input_labels.copy_to(self._init_labels_out)
        
        return sts
        
    def _compute_output_shape(self, inA, inB):
        out_sz = []
        mat_sz = None
        for param in (inA,inB):
            if param._bcip_type == BcipEnums.TENSOR:
                # ensure it is 3D or 2D
                param_rank = len(param.shape)
                if param_rank != 2 and param_rank != 3:
                    print("Both inputs must be either 2D or 3D")
                    return ()
                
                if mat_sz == None:
                    mat_sz = param.shape[-2:]
                elif param.shape[-2:] != mat_sz:
                    return ()
                
                if param_rank == 3:
                    out_sz.append(param.shape[0])
                else:
                    out_sz.append(1)
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
                        return ()
                    
                out_sz.append(param_rank)
            
        return tuple(out_sz)
 
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors or Arrays of Tensors
        for param in (self._inA, self._inB, self._outA):
            if (param._bcip_type != BcipEnums.TENSOR and
                param._bcip_type != BcipEnums.ARRAY):
                return BcipEnums.INVALID_PARAMETERS

        out_sz = self._compute_output_shape(self._inA, self._inB)
        num_combos = out_sz[0]*out_sz[1]
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (self._outA.virtual and len(self._outA.shape) == 0):
            self._outA.shape = out_sz
        
        
        if (self._outA._bcip_type != BcipEnums.TENSOR and
            self._outA.shape != out_sz):
            return BcipEnums.INVALID_PARAMETERS
        elif self._outA._bcip_type == BcipEnums.ARRAY:
            if self._outA.capacity != num_combos:
                return BcipEnums.INVALID_PARAMETERS
            
            for i in range(self._outA.capacity):
                e = self._outA.get_element(i)
                if ((e._bcip_type != BcipEnums.TENSOR and
                     e._bcip_type != BcipEnums.SCALAR) or 
                    (e._bcip_type == BcipEnums.TENSOR and e.shape != (1,1)) or
                    (e._bcip_type == BcipEnums.SCALAR and e.data_type != float)):
                    return BcipEnums.INVALID_PARAMETERS
  
        return BcipEnums.SUCCESS

    def initialization_execution(self):
        """
        Process initialization data if downstream nodes are missing training data
        """
        sts = self.process_data(self._init_inA, self._init_inB, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts
        
    def _process_data(self, inputA, inputB, outputA):
        """
        Execute the kernel and calculate the mean
        """
        def get_obj_data_at_index(obj,index,rank):
            if obj._bcip_type == BcipEnums.TENSOR:
                if rank == 1 and len(obj.shape) == 2:
                    return obj.data
                else:
                    return obj.data[index,:,:]
            else:
                return obj.get_element(index).data
            
        def set_obj_data_at_index(obj,index,data):
            if obj._bcip_type == BcipEnums.TENSOR:
                tensor_data = obj.data # need to extract and edit numpy array b/c tensor currently does not allow sliced modifications
                tensor_data[index] = data
                obj.data = tensor_data
            else:
                e = obj.get_element(index[0]*index[1])
                if e._bcip_type == BcipEnums.TENSOR:
                    e.data = np.asarray([[data]])
                else:
                    e.data = data
        
        out_sz = []
        for in_param in (inputA,inputB):
            if in_param._bcip_type == BcipEnums.TENSOR:
                if len(in_param.shape) == 3:
                    m = in_param.shape[0]
                else:
                    m = 1
            else:
                m = in_param.capacity
        
            out_sz.append(m)
        
        
        for i in range(out_sz[0]):
            # extract the ith element from inA
            x = get_obj_data_at_index(inputA,i,out_sz[0])
            
            for j in range(out_sz[1]):
                # extract the jth element from inB
                y = get_obj_data_at_index(inputB,j,out_sz[1])
                
                try:
                    set_obj_data_at_index(outputA,(i,j),
                                          distance_riemann(x,y))
                
                except:
                    return BcipEnums.FAILURE
                    
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel
        """
        return self.process_data(self._inA, self._inB, self._outA)
    
    @classmethod
    def add_riemann_distance_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a Riemann mean calculating kernel

        Parameters
        ----------
        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor or Array 
            First input data

        inB : Tensor or Array 
            Second Input data

        outA : Tensor or Scalar 
            Output trial data
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

