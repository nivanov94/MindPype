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

        sts = self.train_classifier()
        if sts == BcipEnums.SUCCESS and self._init_outA != None:
            sts = self.initialization_execution()
        
        return sts        

    def train_classifier(self):
        # Setting temporary variables so initialization data not modified between kernels
        temp_labels, temp_tensor = self._labels.data, self._initialization_data.data
        
        if ((not (isinstance(self._initialization_data,Tensor) or 
            isinstance(self._initialization_data, Array))) or 
            (not isinstance(self._labels,Tensor) or 
            isinstance(self._labels, Array))):
                return BcipEnums.INITIALIZATION_FAILURE
        
        # ensure the shpaes are valid
        if len(self._initialization_data.shape) == 3:
            temp_tensor = self._initialization_data.data
            index1, index2, index3 = self._initialization_data.shape
            temp_tensor = np.reshape(temp_tensor, (index1, index2 * index3))

        if len(self._labels.shape) == 2:
            temp_labels = np.squeeze(self._labels.data)
        else:
            temp_labels = self._labels.data

        if (len(temp_tensor.shape) != 2 or len(temp_labels.shape) != 1):
            return BcipEnums.INITIALIZATION_FAILURE
        
        if temp_tensor.shape[0] != temp_labels.shape[0]:
            return BcipEnums.INITIALIZATION_FAILURE

        #print(self._initialization_data.shape, self._labels.shape)
        #print(self._initialization_data.data)
        #all_zeros = not np.any(self._initialization_data.data)
        #print(all_zeros)
        try:
            self._classifier._classifier.fit(temp_tensor, temp_labels)
        except:
            return BcipEnums.INITIALIZATION_FAILURE

        self._initialized = True
        return BcipEnums.SUCCESS


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
        
        return self.process_data(temp_tensor, self._outA)


    def initialization_execution(self):
        """
        Process initialization data. Called if downstream nodes are missing training data
        """

        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data, output_data):
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
    def add_classifier_node(cls, graph, inA, classifier, outA, initialization_data, labels):
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
