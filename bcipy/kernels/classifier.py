from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Scalar

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

        # input must be a tensor or array of tensors
        if (self._inA._bcip_type != BcipEnums.TENSOR and
            self._inA._bcip_type != BcipEnums.ARRAY and
            self._inA._bcip_type != BcipEnums.CIRCLE_BUFFER):
            return BcipEnums.INVALID_PARAMETERS

        if (self._inA._bcip_type != BcipEnums.TENSOR):
            e = self._inA.get_element(0)
            if e._bcip_type != BcipEnums.TENSOR:
                return BcipEnums.INVALID_PARAMETERS

        # output must be scalar, tensor
        if (self._outA._bcip_type != BcipEnums.TENSOR and
            self._outA._bcip_type != BcipEnums.SCALAR)
            return BcipEnums.INVALID_PARAMETERS

        if (self._classifier._bcip_type != BcipEnums.CLASSIFIER):
            return BcipEnums.INVALID_PARAMETERS

        # verify input and output dimensions
        if self._inA._bcip_type == BcipEnums.TENSOR:
            input_sz = self._inA.shape

            if len(input_sz) == 1:
                # single trial/sample mode
                if self._outA._bcip_type == BcipEnums.TENSOR:
                    output_sz = (1,)
                
            elif len(input_sz) == 2:
                #single trial or multi-trial batch mode
                if self._outA._bcip_type == BcipEnums.TENSOR:
                    output_sz = (input_sz[0],)

                elif (self._outA._bcip_type == BcipEnums.SCALAR and
                      input_sz[0] != 1):
                    return BcipEnums.INVALID_PARAMETERS
        else:
            # input is an array
            if (self._outA._bcip_type == BcipEnums.SCALAR):
                return BcipEnums.INVALID_PARAMETERS

            # check elements are correct shape
            e = self._inA.get_element(0)
            input_sz = (self._inA.capacity,) + e.shape

            if len(input_sz) == 2:
                output_sz = (self._inA.capacity,)
            else:
                return BcipEnums.INVALID_PARAMETERS

        if self._outA._bcip_type == BcipEnums.TENSOR:
            if self._outA.virtual and len(self._outA.shape) == 0:
                self._outA.shape = output_sz

            if self._outA.shape != output_sz:
                return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS
        

    def execute(self):
        """
        Execute single trial classification
        """
        # if input is not a tensor, convert
        if self._inA._bcip_type != BcipEnums.TENSOR:
            input_tensor = self._inA.to_tensor()
        else:
            input_tensor = self._inA

        return self._process_data(temp_tensor, self._outA)


    def _process_data(self, input_data, output_data):
        """
        Process data according to outlined kernel function
        """
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        
        out = self._classifier._classifier.predict(input_data.data)

        if output_data._bcip_type == BcipEnums.SCALAR:
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
