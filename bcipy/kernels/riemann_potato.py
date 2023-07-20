from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .kernel_utils import extract_nested_data

import numpy as np

from pyriemann.clustering import Potato
from pyriemann.utils.covariance import covariances
import numpy as np


class RiemannPotatoKernel(Kernel):
    """
    Riemannian potato artifact detection detector

    Parameters
    ----------
   
    graph : Graph 
        Graph that the kernel should be added to

    inputA : Tensor or Array 
        First input data

    outputA : Tensor or Scalar 
        Output trial data

    out_score : 


    """
    
    def __init__(self,graph,inA,outA,thresh,max_iter,regulization,
                 initialization_data=None):
        """
        Kernel takes Tensor input and produces scalar label representing
        the predicted class
        """
        super().__init__('RiemannPotato',BcipEnums.INIT_FROM_DATA,graph)
        self.inputs = [inA]
        self.outputs = [outA]

        self._thresh = thresh
        self._max_iter = max_iter
        self._r = regulization
        
        if initialization_data is not None:
            self.init_inputs = [initialization_data]

        # model will be trained using data in tensor object at later time
        self._initialized = False
        self._potato_filter = None

        
    
    def initialize(self):
        """
        Set reference covariance matrix, mean, and standard deviation
        """
        sts = BcipEnums.SUCCESS
        
        self._initialized = False # clear initialized flag
        sts = self._fit_filter()

        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]

        # compute init output
        if sts == BcipEnums.SUCCESS and init_out is not None and init_in is not None:
            # adjust the shape of init output tensor
            if len(init_in.shape) == 3:
                init_out.shape = (init_in.shape[0],)
 
            sts = self._process_data(init_in, init_out)

            # pass on the labels
            if self.init_input_labels is not None:
                self.copy_init_labels_to_output()

        if sts == BcipEnums.SUCCESS:
            self._initialized = True
        
        return sts
        
       
    def _fit_filter(self):
        """
        fit the potato filter using the initialization data
        """
        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]

        if (init_in.bcip_type != BcipEnums.TENSOR and
            init_in.bcip_type != BcipEnums.ARRAY  and
            init_in.bcip_type != BcipEnums.CIRCLE_BUFFER):
            return BcipEnums.INITIALIZATION_FAILURE
        
        if init_in.bcip_type == BcipEnums.TENSOR: 
            X = init_in.data
        else:
            try:
                # extract the data from a potentially nested array of tensors
                X = extract_nested_data(init_in)
            except:
                return BcipEnums.INITIALIZATION_FAILURE
            
        if len(X.shape) != 3:
            return BcipEnums.INITIALIZATION_FAILURE
        
        if X.shape[-2] != X.shape[-1]:
            # convert to covs
            X = covariances(X)
            X = (1-self._r)*X + self._r*np.eye(X.shape[-1])
        
        self._potato_filter = Potato(threshold=self._thresh, n_iter_max=self._max_iter)
        self._potato_filter.fit(X)

        return BcipEnums.SUCCESS

    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """

        d_in = self.inputs[0]
        d_out = self.outputs[0]

        # first ensure the input is a tensor
        if d_in.bcip_type != BcipEnums.TENSOR:
            return BcipEnums.INVALID_PARAMETERS

        # ensure the output is a tensor or scalar
        if (d_out.bcip_type != BcipEnums.TENSOR and
            d_out.bcip_type != BcipEnums.SCALAR):
            return BcipEnums.INVALID_PARAMETERS

        # check thresh and max iterations
        if self._thresh < 0 or self._max_iter < 0:
            return BcipEnums.INVALID_PARAMETERS

        # check in/out dimensions        
        input_shape = d_in.shape
        input_rank = len(input_shape)
        
        # input tensor should not be greater than rank 3
        if input_rank > 3 or input_rank < 2:
            return BcipEnums.INVALID_PARAMETERS
        
        # input should be a covariance matrix
        if input_shape[-2] != input_shape[-1]:
            return BcipEnums.INVALID_PARAMETERS
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (d_out.bcip_type == BcipEnums.TENSOR and 
            d_out.virtual and
            len(d_out.shape) == 0):
            if input_rank == 2:
                d_out.shape = (1,)
            else:
                d_out.shape = (input_shape[0],)
        
        # check for dimensional alignment
        if d_out.bcip_type == BcipEnums.SCALAR:
            # input tensor should only be a single trial
            if len(d_in.shape) == 3:
                # first dimension must be equal to one
                if d_in.shape[0] != 1:
                    return BcipEnums.INVALID_PARAMETERS
        else:
            # check that the dimensions of the output match the dimensions of
            # input
            if d_in.shape[0] != d_out.shape[0]:
                return BcipEnums.INVALID_PARAMETERS

            # output tensor should be one dimensional
            if len(d_out.shape) > 1:
                return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS


    def _process_data(self, inA, outA):
        input_data = inA.data
        if len(inA.shape) == 2:
            # pyriemann library requires input data to have 3 dimensions with the 
            # first dimension being 1
            input_data = input_data[np.newaxis,:,:]

        try:
            input_data = (1-self._r)*input_data + self._r*np.eye(inA.shape[-1])
            output = self._potato_filter.predict(input_data)

            if outA.bcip_type == BcipEnums.SCALAR:
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
            return self._process_data(self.inputs[0], self.outputs[0])

       
    @classmethod
    def add_riemann_potato_node(cls,graph,inA,outA, initialization_data=None,
                                thresh=3,max_iter=100,regularization=0.01):
        """
        Factory method to create a riemann potato artifact detector
        """
        
        # create the kernel object            

        k = cls(graph,inA,outA,thresh,max_iter,regularization,
                initialization_data)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA, BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
