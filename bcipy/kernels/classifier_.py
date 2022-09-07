from classes.kernel import Kernel
from classes.node import Node 
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.array import Array
from classes.classifier import Classifier
from classes.bcip_enums import BcipEnums

import numpy as np

class ClassifierKernel(Kernel):
    """
    Classify data using BCIP Classifier Object

    Parameters
    ----------
    
    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor object (n_channels, n_samples)
        - Input trial data

    classifier : Classifier object
        - BCIP Classifier object to be used for classification

    outA : Scalar object
        - Output trial data

    initialization_data : Tensor object, (n_trials, n_channels, n_samples)
        - Initialization data to train the classifier
    
    labels : Tensor object, (n_trials, )
        - Labels corresponding to initialization data class labels 
        - (n_trials, 2) for class separated data where column 1 is the trial label and column 2 is the start index

    """

    def __init__(self, graph, inA, classifier, outA, initialization_data, labels):
        super().__init__('Classifier', BcipEnums.INIT_FROM_DATA, graph)
        self._inA = inA
        self._classifier = classifier
        self._outA = outA
        self._initialization_data = initialization_data 
        self._labels = labels


        self._initialized = False
        self._init_inA = None
        self._init_outA = None


    def initialize(self):

        if self._initialization_data == None:
            self._initialization_data = self._init_inA

        sts = BcipEnums.SUCCESS
        
        if ((self._initialization_data._bcip_type != BcipEnums.TENSOR and 
             self._initialization_data._bcip_tpe != BcipEnums.ARRAY and
             self._initialization_data.bcip_type != BcipEnums.CIRCLE_BUFFER) or
            (self._labels._bcip_type != BcipEnums.TENSOR and 
             self._labels._bcip_type != BcipEnums.ARRAY and
             self._labels.bcip_type != BcipEnums.CIRCLE_BUFFER)):
            return BcipEnums.INITIALIZATION_FAILURE
        
        # if data or labels are in an array, extract them to a tensor
        if self._initialization_data._bcip_type == BcipEnums.TENSOR:
            local_init_data = self._initialization_data.data
        else:
            local_init_tensor = self._initialization_data.to_tensor()
            local_init_data = local_init_tensor.data

        if self._labels._bcip_type == BcipEnums.TENSOR:
            local_init_labels = self._labels.data
        else:
            local_init_labels = self._labels.to_tensor().data
        
        # ensure the shapes are valid
        if len(local_init_data.shape) == 3:
            index1, index2, index3 = local_init_data.shape
            local_init_data = np.reshape(local_init_data, (index1, index2 * index3))

        if len(local_init_labels.shape) == 2:
            local_init_labels = np.squeeze(local_init_labels)


        if (len(local_init_data.shape) != 2 or len(local_init_labels.shape) != 1):
            return BcipEnums.INITIALIZATION_FAILURE
        
        if local_init_data.shape[0] != local_init_labels.shape[0]:
            return BcipEnums.INITIALIZATION_FAILURE


        try:
            self._classifier._classifier.fit(local_init_data, local_init_labels)
        except:
            return BcipEnums.INITIALIZATION_FAILURE

        self._initialized = True
        
        if sts == BcipEnums.SUCCESS and self._init_outA != None:
            sts = self._process_data(local_init_tensor, self._init_outA)
        
        return sts
        

    def verify(self):
        """similar verification process to individual classifier kernels"""

        if ((not isinstance(self._inA, Tensor)) or
            (not isinstance(self._outA, Scalar)) or
            (not isinstance(self._classifier, Classifier))):
            return BcipEnums.INVALID_PARAMETERS
        input_shape = self._inA.shape
        input_rank = len(input_shape)

        if input_rank == 0:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS
        

    def execute(self):
        """
        Execute single trial classification
        """
        
        if len(self._inA.shape) == 2:
            temp_input = np.reshape(self._inA.data, (1, self._inA.shape[0]*self._inA.shape[1]))
            temp_tensor = Tensor.create_from_data(self._session, np.shape(temp_input), temp_input)
        
        return self._process_data(temp_tensor, self._outA)


    def _process_data(self, input_data, output_data):
        """
        Process data according to outlined kernel function
        """
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        
        out = self._classifier._classifier.predict(input_data.data)

        if isinstance(output_data,Scalar):
            output_data.data = int(out)
        else:
            output_data.data = out
        
        return BcipEnums.SUCCESS


    @classmethod
    def add_classifier_node(cls, graph, inA, classifier, outA, initialization_data = None, labels = None):
        """
        Factory method to create a classifier kernel and add it to a graph as a generic node object
        
        Parameters
        ----------

        graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor object (n_channels, n_samples)
            - Input trial data

        classifier : Classifier object
            - BCIP Classifier object to be used for classification

        outA : Scalar object
            - Output trial data

        initialization_data : Tensor object, (n_trials, n_channels, n_samples)
            - Initialization data to train the classifier
        
        labels : Tensor object, (n_trials, )
            - Labels corresponding to initialization data class labels 
            - (n_trials, 2) for class separated data where column 1 is the trial label and column 2 is the start index
        """

        #create the kernel object
        c = cls(graph, inA, classifier, outA, initialization_data, labels)

        params = (Parameter(inA, BcipEnums.INPUT),
                  Parameter(outA, BcipEnums.OUTPUT))

        node = Node(graph, c, params)

        graph.add_node(node)

        return node
