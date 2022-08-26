"""
Created on Fri Nov 22 09:12:33 2019

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.array import Array
from classes.circle_buffer import CircleBuffer
from classes.bcip_enums import BcipEnums

import numpy as np

from pyriemann import classification


# TODO make this available for all classifier functions
def _extract_nested_data(bcip_obj):
    """
    Recursively extract Tensor data within a BCIP array or array-of-arrays
    """
    if not isinstance(bcip_obj,Array):
        return np.array(())
    
    N = bcip_obj.num_elements
    
    X = None
    for i in range(N):
        e = bcip_obj.get_element(i)
        if isinstance(e,Tensor):
            elem_data = np.expand_dims(e.data,0) # add dimension so we can append below
        elif isinstance(e,Scalar):
            elem_data = np.asarray([e.data])
        else:
            elem_data = _extract_nested_data(e)
        
        if X is None:
            X = elem_data
        else:
            X = np.append(X,elem_data,axis=0)
    
    return X

class RiemannMDMClassifierKernel(Kernel):
    """
    Riemannian Minimum Distance to the Mean Classifier. Kernel takes Tensor input and produces scalar label representing
    the predicted class. Review classmethods for specific input parameters

    Parameters
    ----------
    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor or Array object
        - First input data

    outA : Tensor or Scalar object
        - Output trial data

    init_style : BcipEnums Object
        - Indicates the type of classifier kernel (based by class functions)

    initialize_params : dict
        - Object passed by classmethods that contains training data and training labels 
    """
    
    def __init__(self,graph,inputA,outputA,init_style,initialize_params):
        """
        Kernel takes Tensor input and produces scalar label representing
        the predicted class
        """
        super().__init__('RiemannMDM',init_style,graph)
        self._inputA  = inputA
        self._outputA = outputA

        self._init_inA = initialize_params['initialization_data']
        self._init_outA = None
        
        self._init_params = initialize_params
        
        if init_style == BcipEnums.INIT_FROM_DATA:
            # model will be trained using data in tensor object at later time
            self._initialized = False
            self._classifier = None
        elif init_style == BcipEnums.INIT_FROM_COPY:
            # model is copy of predefined MDM model object
            self._classifier = initialize_params['model']
            self._initialized = True
        
        self._graph = graph
    
    def initialize(self):
        """
        Set the means for the classifier
        """
        
        if self.init_style == BcipEnums.INIT_FROM_DATA:
            return self.train_classifier()
        else:
            # kernel contains a reference to a pre-existing MDM object, no
            # need to train here
            self._initialized = True
            return BcipEnums.SUCCESS
        
    def train_classifier(self):
        """
        Train the classifier
        
        The method will update the kernel's internal representation of the
        classifier
        """
        
        if (not (isinstance(self._init_params['initialization_data'],Tensor) or 
                 isinstance(self._init_params['initialization_data'],Array)) or 
            not (isinstance(self._init_params['labels'],Tensor) or
                 isinstance(self._init_params['labels'],Array))):
                return BcipEnums.INITIALIZATION_FAILURE
        
        if isinstance(self._init_params['initialization_data'],Tensor): 
            X = self._init_params['initialization_data'].data
        else:
            try:
                # extract the data from a potentially nested array of tensors
                X = _extract_nested_data(self._init_params['initialization_data'])
            except:
                return BcipEnums.INITIALIZATION_FAILURE
            
        if isinstance(self._init_params['labels'],Tensor):
            y = self._init_params['labels'].data
        else:
            y = _extract_nested_data(self._init_params['labels'])
        
        # ensure the shpaes are valid
        if len(X.shape) != 3 or len(y.shape) != 1:
            return BcipEnums.INITIALIZATION_FAILURE
        
        if X.shape[0] != y.shape[0]:
            return BcipEnums.INITIALIZATION_FAILURE
        
        self._classifier = classification.MDM()
        self._classifier.fit(X,y)
        
        self._initialized = True
        
        return BcipEnums.SUCCESS
    
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """

        if self._init_params['initialization_data'] == None:
            self._graph._missing_data = True
        
        # first ensure the input and output are tensors
        if (not isinstance(self._inputA,Tensor)) or \
            (not (isinstance(self._outputA,Tensor) or 
                  isinstance(self._outputA,Scalar))):
                return BcipEnums.INVALID_PARAMETERS
        
        input_shape = self._inputA.shape
        input_rank = len(input_shape)
        
        # input tensor should not be greater than rank 3
        if input_rank > 3 or input_rank < 2:
            return BcipEnums.INVALID_PARAMETERS
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (isinstance(self._outputA,Tensor) and self._outputA.virtual \
            and self._outputA.shape == None):
            if input_rank == 2:
                self._outputA.shape = (1,)
            else:
                self._outputA.shape = (input_shape[0],)
        
        
        # check for dimensional alignment
        
        if isinstance(self._outputA,Scalar):
            # input tensor should only be a single trial
            if len(self._inputA.shape) == 3:
                # first dimension must be equal to one
                if self._inputA.shape[0] != 1:
                    return BcipEnums.INVALID_PARAMETERS
        else:
            # check that the dimensions of the output match the dimensions of
            # input
            if self._inputA.shape[0] != self._outputA.shape[0]:
                return BcipEnums.INVALID_PARAMETERS

            # output tensor should be one dimensional
            if len(self._outputA.shape) > 1:
                return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel by classifying the input trials
        """
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        
        # pyriemann library requires input data to have 3 dimensions with the 
        # first dimension being 1
        input_data = self._inputA.data
        input_data = input_data[np.newaxis,:,:]
        
        if isinstance(self._outputA,Tensor):
            self._outputA.data = self._classifier.predict(input_data)
        else:
            self._outputA.data = int(self._classifier.predict(input_data))
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_untrained_riemann_MDM_node(cls,graph,inputA,outputA,\
                                       initialization_data,labels):
        """
        Factory method to create an untrained riemann minimum distance 
        to the mean classifier kernel and add it to a graph
        as a generic node object.
        
        Note that the node will have to be initialized (i.e. trained) prior 
        to execution of the kernel.

        Parameters
        ----------
        graph : Graph Object
            - Graph that the kernel should be added to

        inputA : Tensor or Array object
            - First input data

        outputA : Tensor or Scalar object
            - Output trial data

        initialization_data : BCIPy Tensor Object
            - Initialization data to train the classifier with (n_trials, n_channels, n_samples)

        labels : BCIPy Tensor Object
            - Class labels for initialization data (n_trials,)
        """
        
        # create the kernel object            
        init_params = {'initialization_data' : initialization_data, 
                       'labels'        : labels}
        k = cls(graph,inputA,outputA,BcipEnums.INIT_FROM_DATA,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT), \
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
    
    @classmethod
    def add_trained_riemann_MDM_node(cls,graph,inputA,outputA,\
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
        graph : Graph Object
            - Graph that the kernel should be added to

        inputA : Tensor or Array object
            - First input data

        outputA : Tensor or Scalar object
            - Output trial data

        model : MDM Classifier object
            - Existing MDM Classifier object that will be added to the node (must be pre-trained)
        """

        # sanity check that the input is actually an MDM model
        if not isinstance(model,classification.MDM):
            return None
        
        # create the kernel object
        init_params = {'model' : model}
        k = cls(graph,inputA,outputA,BcipEnums.INIT_FROM_COPY,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT), \
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
