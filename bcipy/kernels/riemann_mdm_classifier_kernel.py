from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .kernel_utils import extract_nested_data

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

    init_style : BcipEnums 
        Indicates the type of classifier kernel (based by class functions)

    initialize_params : dict
        Object passed by classmethods that contains training data and training labels 
    """
    
    def __init__(self,graph,inA,outA,init_style,init_params):
        """
        Kernel takes Tensor input and produces scalar label representing
        the predicted class
        """
        super().__init__('RiemannMDM',init_style,graph)
        self._inA  = inA
        self._outA = outA

        self._init_params = init_params

        if 'initialization_data' in init_params:
            self._init_inA = init_params['initialization_data']
        else:
            self._init_inA = None

        if 'labels' in init_params:
            self._init_labels_in = init_params['labels']
        else:
            self._init_labels_in = None

        self._init_outA = None
        self._init_labels_out = None
 
        if init_style == BcipEnums.INIT_FROM_DATA:
            # model will be trained using data in tensor object at later time
            self._initialized = False
            self._classifier = None
        elif init_style == BcipEnums.INIT_FROM_COPY:
            # model is copy of predefined MDM model object
            self._classifier = init_params['model']
            self._initialized = True
        
    
    def initialize(self):
        """
        Set the means for the classifier
        """
        sts = BcipEnums.SUCCESS
        
        if self.init_style == BcipEnums.INIT_FROM_DATA:
            self._initialized = False # clear initialized flag
            sts = self.train_classifier()

        # compute init output
        if sts == BcipEnums.SUCCESS and self._init_outA is not None:
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
 
        
    def train_classifier(self):
        """
        Train the classifier
        
        The method will update the kernel's internal representation of the
        classifier
        """
        
        if ((self._init_inA._bcip_type != BcipEnums.TENSOR and
             self._init_inA._bcip_type != BcipEnums.ARRAY)  or
            (self._init_labels_in._bcip_type != BcipEnums.TENSOR and
             self._init_labels_in._bcip_type != BcipEnums. ARRAY)):
                return BcipEnums.INITIALIZATION_FAILURE
        
        if self._init_inA._bcip_type == BcipEnums.TENSOR: 
            X = self._init_params['initialization_data'].data
        else:
            try:
                # extract the data from a potentially nested array of tensors
                X = extract_nested_data(self._init_inA)
            except:
                return BcipEnums.INITIALIZATION_FAILURE
            
        if self._init_labels_in._bcip_type == BcipEnums.TENSOR:
            y = self._init_labels_in.data
        else:
            y = extract_nested_data(self._init_labels_in)
        
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

        # first ensure the input is a tensor
        if self._inA._bcip_type != BcipEnums.TENSOR:
            return BcipEnums.INVALID_PARAMETERS

        # ensure the output is a tensor or scalar
        if (self._outA._bcip_type != BcipEnums.TENSOR and
            self._outA._bcip_type != BcipEnums.SCALAR):
            return BcipEnums.INVALID_PARAMETERS
        
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
            output = self._classifier.predict(input_data)

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
    def add_untrained_riemann_MDM_node(cls,graph,inA,outA,
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
        init_params = {'initialization_data' : initialization_data, 
                       'labels'        : labels}
        k = cls(graph,inA,outA,BcipEnums.INIT_FROM_DATA,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT), 
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
    
    @classmethod
    def add_trained_riemann_MDM_node(cls,graph,inA,outA,
                                     model):
        """
        Factory method to create a riemann minimum distance 
        to the mean classifier kernel containing a copy of a pre-trained
        MDM classifier and add it to a graph as a generic node object.
        
        The kernel will contain a reference to the model rather than making a 
        deep-copy. Therefore any changes to the classifier object outside
        will effect the classifier here.

        Parameters
        ----------
        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor or Array 
            First input data

        outA : Tensor or Scalar 
            Output trial data

        model : RiemannMDMClassifierKernel
            Existing MDM Classifier object that will be added to the node (must be pre-trained)
        """

        # sanity check that the input is actually an MDM model
        if not isinstance(model,classification.MDM):
            return None
        
        # create the kernel object
        init_params = {'model' : model}
        k = cls(graph,inA,outA,BcipEnums.INIT_FROM_COPY,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
