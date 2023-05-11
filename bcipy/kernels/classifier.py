from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor
from .kernel_utils import extract_nested_data


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

    return_probabilities : bool, default = False
        - If True, the output will be the probability of each class instead of the class label.
        - Probablities are returned with shape (n_samples, n_classes).
            - For single trial classification, the shape will be (1, n_classes)
    """

    def __init__(self, graph, inA, classifier, outA, initialization_data, labels, return_probabilities):
        super().__init__('Classifier', BcipEnums.INIT_FROM_DATA, graph)
        self._inA = inA
        self._classifier = classifier
        self._outA = outA
        
        self._initialized = False
        self._init_inA = initialization_data
        self._init_outA = None
        self._init_labels_in = labels
        self.return_probabilities = return_probabilities


    def initialize(self):

        sts = BcipEnums.SUCCESS
        
        if ((self._init_inA._bcip_type != BcipEnums.TENSOR and
             self._init_inA._bcip_type != BcipEnums.ARRAY  and
             self._init_inA._bcip_type != BcipEnums.CIRCLE_BUFFER) or
            (self._init_labels_in._bcip_type != BcipEnums.TENSOR and
             self._init_labels_in._bcip_type != BcipEnums.ARRAY  and
             self._init_labels_in._bcip_type != BcipEnums.CIRCLE_BUFFER)):
            return BcipEnums.INITIALIZATION_FAILURE
        
        
        if self._init_inA._bcip_type == BcipEnums.TENSOR: 
            X = self._init_inA.data
        else:
            try:
                # extract the data from a potentially nested array of tensors
                X = extract_nested_data(self._init_inA)
            except:
                return BcipEnums.INITIALIZATION_FAILURE    
        
        if self._init_labels_in._bcip_type == BcipEnums.TENSOR:    
            y = self._init_labels_in.data
        else:
            try:
                y = extract_nested_data(self._init_labels_in)
            except:
                return BcipEnums.INITIALIZATION_FAILURE
        
        
        # ensure the shapes are valid
        if len(X.shape) == 3:
            index1, index2, index3 = X.shape
            X = np.reshape(X, (index1, index2 * index3))

        if len(y.shape) == 2:
            y = np.squeeze(y)


        if (len(X.shape) != 2 or len(y.shape) != 1):
            return BcipEnums.INITIALIZATION_FAILURE
        
        if X.shape[0] != y.shape[0]:
            return BcipEnums.INITIALIZATION_FAILURE


        try:
            self._classifier._classifier.fit(X, y)
        except:
            return BcipEnums.INITIALIZATION_FAILURE

        self._initialized = True
        
        if sts == BcipEnums.SUCCESS and self._init_outA != None:
            init_tensor = Tensor.create_from_data(self.session, X.shape, X)
            sts = self._process_data(init_tensor, self._init_outA)

            # pass on the labels
            if self._init_labels_in._bcip_type != BcipEnums.TENSOR:
                input_labels = self._init_labels_in.to_tensor()
            else:
                input_labels = self._init_labels_in
            input_labels.copy_to(self._init_labels_out)
        
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
            self._outA._bcip_type != BcipEnums.SCALAR):
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

        return self._process_data(input_tensor, self._outA)


    def _process_data(self, inA, outA):
        """
        Process data according to outlined kernel function
        """
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        
        if len(inA.shape) == 1:
            input_data = np.expand_dims(inA.data,axis=0)
        else:
            input_data = inA.data
        
        if self.return_probabilities:
            output_data = self._classifier._classifier.predict_proba(input_data)
            try:
                outA.data = output_data
            except Exception as e:
                print(e)
                return BcipEnums.EXE_FAILURE

        else:
            output_data = self._classifier._classifier.predict(input_data)

            if outA._bcip_type == BcipEnums.SCALAR:
                outA.data = int(output_data)
            else:
                outA.data = output_data
        
        return BcipEnums.SUCCESS


    @classmethod
    def add_classifier_node(cls, graph, inA, classifier, outA, initialization_data = None, labels = None, return_probabilities = False):
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
        
        return_probabilities : bool, default = False
            - If True, the output will be the probability of each class instead of the class label.
            - Probablities are returned with shape (n_samples, n_classes).
                - For single trial classification, the shape will be (1, n_classes)
        """

        #create the kernel object
        c = cls(graph, inA, classifier, outA, initialization_data, labels, return_probabilities)

        params = (Parameter(inA, BcipEnums.INPUT),
                  Parameter(outA, BcipEnums.OUTPUT))

        node = Node(graph, c, params)

        graph.add_node(node)

        return node
