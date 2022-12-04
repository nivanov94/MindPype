from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .kernel_utils import extract_nested_data

from ..containers import Scalar

import numpy as np

from pyriemann.clustering import Potato
import numpy as np


class RiemannPotatoKernel(Kernel):
    """
    Riemannian potato artifact detection detector

    Parameters
    ----------
   
    graph : Graph Object
        - Graph that the kernel should be added to

    inputA : Tensor or Array object
        - First input data

    outputA : Tensor or Scalar object
        - Output trial data

    out_score : 


    """
    
    def __init__(self,graph,inA,out_label,thresh,max_iter,init_style,
                 initialization_data):
        """
        Kernel takes Tensor input and produces scalar label representing
        the predicted class
        """
        super().__init__('RiemannPotato',init_style,graph)
        self._inA  = inA
        self._outA = out_label

        self._thresh = thresh
        self._max_iter = max_iter
        

        self._init_inA = initialization_data
        self._init_labels_in = None
        self._init_labels_out = None
        self._init_outA = None
 
        # model will be trained using data in tensor object at later time
        self._initialized = False
        self._potato_filter = None

        
    
    def initialize(self):
        """
        Set reference covariance matrix, mean, and standard deviation
        """
        sts = BcipEnums.SUCCESS
        
        if self.init_style == BcipEnums.INIT_FROM_DATA:
            self._initialized = False # clear initialized flag
            sts = self._fit_filter()

        # compute init output
        if sts == BcipEnums.SUCCESS and self._init_outA != None:
            # adjust the shape of init output tensor
            if len(self._init_inA.shape) == 3:
                self._init_outA.shape = (self._init_inA.shape[0],)
 
            sts = self._process_data(self._init_inA, self._init_outA)

            # pass on the labels
            if self._init_labels_in._bcip_type != BcipEnums.TENSOR:
                input_labels = self._init_labels_in.to_tensor()
            else:
                input_labels = self._init_labels_in
            input_labels.copy_to(self._init_labels_out)

        if sts == BcipEnums.SUCCESS:
            self._initialized = True
        
        return sts
        
       
    def _fit_filter(self):
        """
        fit the potato filter using the initialization data
        """

        if (self._init_inA._bcip_type != BcipEnums.TENSOR and
            self._init_inA._bcip_type != BcipEnums.ARRAY):
            return BcipEnums.INITIALIZATION_FAILURE
        
        if self._init_inA._bcip_type == BcipEnums.TENSOR: 
            X = self._init_params['initialization_data'].data
        else:
            try:
                # extract the data from a potentially nested array of tensors
                X = extract_nested_data(self._init_inA)
            except:
                return BcipEnums.INITIALIZATION_FAILURE
            
        if len(X.shape) != 3:
            return BcipEnums.INITIALIZATION_FAILURE
        
        self._potato_filter = Potato(thresh=self._thresh, n_iter_max=self._max_iter)
        self._potato_filter.fit(X)

        return BcipEnums.SUCCESS

    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        # first ensure the input is a tensor
        if self._inA._bcip_type != BcipEnums.TENSOR:
            return BcipEnums.INVALID_PARAMETERS

        # ensure the output is a tensor or scalar
        if (self._outA._bcip_type != BcipEnums.TENSOR and
            self._outA._bcip_type != BcipEnums.SCALAR):
            return BcipEnums.INVALID_PARAMETERS

        # check thresh and max iterations
        if self._thresh < 0 or self._max_iter < 0:
            return BcipEnums.INVALID_PARAMETERS

        # check in/out dimensions        
        input_shape = self._inA.shape
        input_rank = len(input_shape)
        
        # input tensor should not be greater than rank 3
        if input_rank > 3 or input_rank < 2:
            return BcipEnums.INVALID_PARAMETERS
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (self._outA._bcip_type == BcipEnums.TENSOR and 
            self._outA.virtual and
            len(self._outA.shape) == 0):
            if input_rank == 2:
                self._outA.shape = (1,)
            else:
                self._outA.shape = (input_shape[0],)
        
        # check for dimensional alignment
        if self._outA._bcip_type == BcipEnums.SCALAR:
            # input tensor should only be a single trial
            if len(self._inA.shape) == 3:
                # first dimension must be equal to one
                if self._inA.shape[0] != 1:
                    return BcipEnums.INVALID_PARAMETERS
        else:
            # check that the dimensions of the output match the dimensions of
            # input
            if self._inA.shape[0] != self._outA.shape[0]:
                return BcipEnums.INVALID_PARAMETERS

            # output tensor should be one dimensional
            if len(self._outA.shape) > 1:
                return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS



    def _process_data(self, inA, outA):
        input_data = inA.data
        if len(inA.shape) == 2:
            # pyriemann library requires input data to have 3 dimensions with the 
            # first dimension being 1
            input_data = input_data[np.newaxis,:,:]

        try:
            output = self._potato_filter.predict(input_data)

            if outA._bcip_type == BcipEnums.SCALAR:
                outA.data = int(output)
            else:
                outA.data = output

            return BcipEnums.SUCCESS

        except:
            return BcipEnums.EXE_FAILURE
        
   
 
    def execute(self):
        """
        Execute the kernel by classifying the input trials
        """
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        else:
            return self._process_data(self._inA, self._outA)

       
    @classmethod
    def add_riemann_potato_node(cls,graph,inA,initialization_data,
                                out_label,
                                thresh=3,max_iter=100):
        """
        Factory method to create a riemann potato artifact detector
        """
        
        # create the kernel object            

        k = cls(graph,inA,out_labels,out_scores,thresh,max_iter,
                initialization_data)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(out_label, BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

