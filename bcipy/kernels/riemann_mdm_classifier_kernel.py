from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .kernel_utils import extract_init_inputs

from ..containers import Scalar

import numpy as np

from pyriemann import classification


class RiemannMDMClassifierKernel(Kernel):
    """
    Riemannian Minimum Distance to the Mean Classifier. Kernel takes Tensor input and produces scalar label representing
    the predicted class. Review classmethods for specific input parameters

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor or Array 
        First input data

    outA : Tensor or Scalar 
        Output trial data

    initialization_data : Tensor
        Initialization data to train the classifier (n_trials, n_channels, n_samples)
    
    labels : Tensor
        Labels corresponding to initialization data class labels (n_trials, )
        (n_trials, 2) for class separated data where column 1 is the trial label and column 2 is the start index

    """
    
    def __init__(self,graph,inA,outA,initialization_data,labels):
        """
        Kernel takes Tensor input and produces scalar label representing
        the predicted class
        """
        super().__init__('RiemannMDM',BcipEnums.INIT_FROM_DATA,graph)
        self.inputs = [inA]
        self.outputs = [outA]

        self._initialized = False

        if initialization_data is not None:
            self.init_inputs = [initialization_data]

        if labels is not None:
            self.init_input_labels = labels
    
    def initialize(self):
        """
        Set the means for the classifier
        """
        sts = BcipEnums.SUCCESS
        
        self._initialized = False # clear initialized flag
        sts = self.train_classifier()

        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]

        # compute init output
        if sts == BcipEnums.SUCCESS and init_out is not None:
            # adjust the shape of init output tensor
            if len(init_in.shape) == 3:
                init_out.shape = (init_in.shape[0],)

            # compute the init output
            sts = self._process_data(init_in, init_out)

            # pass on the labels
            self.copy_init_labels_to_output()

        if sts == BcipEnums.SUCCESS:
            self._initialized = True
        
        return sts


    def train_classifier(self):
        """
        Train the classifier
        
        The method will update the kernel's internal representation of the
        classifier
        """

        init_in = self.init_inputs[0]
        init_labels = self.init_input_labels

        # check that the input data is valid
        if ((init_in.bcip_type != BcipEnums.TENSOR and
             init_in.bcip_type != BcipEnums.ARRAY)  or
            (init_labels.bcip_type != BcipEnums.TENSOR and
             init_labels.bcip_type != BcipEnums. ARRAY)):
                return BcipEnums.INITIALIZATION_FAILURE
        
        # extract the initialiation data
        X = extract_init_inputs(init_in)
        y = extract_init_inputs(init_labels)
            
        # ensure the shpaes are valid
        if len(X.shape) != 3 or len(y.shape) != 1:
            return BcipEnums.INITIALIZATION_FAILURE
        
        if X.shape[0] != y.shape[0]:
            return BcipEnums.INITIALIZATION_FAILURE
        
        self._classifier = classification.MDM()
        self._classifier.fit(X,y)
        
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
        
        input_shape = d_in.shape
        input_rank = len(input_shape)
        
        # input tensor should not be greater than rank 3
        if input_rank > 3 or input_rank < 2:
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
            output = self._classifier.predict(input_data)

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
    def add_riemann_MDM_node(cls,graph,inA,outA,
                             initialization_data=None,labels=None):
        """
        Factory method to create an untrained riemann minimum distance 
        to the mean classifier kernel and add it to a graph
        as a generic node object.
        
        Note that the node will have to be initialized (i.e. trained) prior 
        to execution of the kernel.

        Parameters
        ----------
        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor or Array 
            First input data

        outA : Tensor or Scalar 
            Output trial data

        initialization_data : Tensor 
            Initialization data to train the classifier with (n_trials, n_channels, n_samples)

        labels : Tensor 
            Class labels for initialization data (n_trials,)
        """
        
        # create the kernel object            
        k = cls(graph,inA,outA,initialization_data,labels)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT), 
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    