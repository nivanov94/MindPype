from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor
from .kernel_utils import extract_init_inputs


import numpy as np
import warnings

class ClassifierKernel(Kernel):
    """
    Classify data using BCIP Classifier Object

    Parameters
    ----------
    
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor
        Input trial data (n_channels, n_samples)

    classifier : Classifier 
        BCIP Classifier object to be used for classification

    Prediction : Scalar 
        Classifier prediction

    output_probs : Tensor
        If not None, the output will be the probability of each class. Default is None.
        
    initialization_data : Tensor
        Initialization data to train the classifier (n_trials, n_channels, n_samples)
    
    labels : Tensor
        Labels corresponding to initialization data class labels (n_trials, )
        (n_trials, 2) for class separated data where column 1 is the trial label and column 2 is the start index
    """

    def __init__(self, graph, inA, classifier, prediction, output_probs, initialization_data = None, labels = None):
        super().__init__('Classifier', BcipEnums.INIT_FROM_DATA, graph)
        self.inputs = [inA]
        self._classifier = classifier
        self.outputs = [prediction, output_probs]
        
        self._initialized = False

        if initialization_data is not None:
            self.init_inputs = [initialization_data]

        if labels is not None:
            self.init_input_labels = labels


    def _initialize(self, init_inputs, init_outputs, labels):

        # check that the input init data is in the correct type
        init_in = init_inputs[0]
        accepted_inputs = (BcipEnums.TENSOR,BcipEnums.ARRAY,BcipEnums.CIRCLE_BUFFER)
        
        for init_obj in (init_in,labels):
            if init_obj.bcip_type not in accepted_inputs:
                raise TypeError('Initialization data must be a tensor or array of tensors')
    
        # extract the initialization data from a potentially nested array of tensors 
        X = extract_init_inputs(init_in)
        y = extract_init_inputs(labels)

        # ensure the shapes are valid
        if len(X.shape) == 3:
            index1, index2, index3 = X.shape
            X = np.reshape(X, (index1, index2 * index3))

        if len(y.shape) == 2:
            y = np.squeeze(y)

        # initialize the classifier
        self._classifier._classifier.fit(X, y)

        # set the initialization output        
        if init_outputs[0] is not None or init_outputs[1] is not None:
            init_tensor = Tensor.create_from_data(self.session, X.shape, X)

            # adjust output shapes if necessary
            if self.init_outputs[0] is not None and self.init_outputs[0].virtual:
                self.init_outputs[0].shape = (X.shape[0],)
            
            if self.init_outputs[1] is not None and self.init_outputs[1].virtual:
                self.init_outputs[1].shape = (X.shape[0], self._classifier.n_classes)

            self._process_data([init_tensor], 
                               self.init_outputs) 


    def _verify(self):
        """similar verification process to individual classifier kernels"""

        # inputs must be a tensor or array of tensors
        accepted_input_types = (BcipEnums.TENSOR, 
                                BcipEnums.ARRAY, 
                                BcipEnums.CIRCLE_BUFFER)
        
        d_in = self.inputs[0]
        if d_in.bcip_type not in accepted_input_types:
            raise TypeError('Input data must be a tensor or array of tensors')

        # if input is an array, check that its elements are tensors
        if (d_in.bcip_type != BcipEnums.TENSOR):
            e = d_in.get_element(0)
            if e.bcip_type != BcipEnums.TENSOR:
                raise TypeError('Input data must be a tensor or array of tensors')

        # check that the classifier is valid
        if (self._classifier.bcip_type != BcipEnums.CLASSIFIER):
            raise TypeError('Classifier must be a BCIP Classifier object')
        
        # ensure the classifier has a predict method
        if (not hasattr(self._classifier._classifier, 'predict') or 
            not callable(self._classifier._classifier.predict)):
            raise Exception('Classifier does not have a predict method')
        
        # if using probability output, ensure the classifier has a predict_proba method
        if (self.outputs[1] is not None and
            (not hasattr(self._classifier._classifier, 'predict_proba') or
             not callable(self._classifier._classifier.predict_proba))):
            raise Exception('Classifier does not have a predict_proba method')


    def _process_data(self, inputs, outputs):
        """
        Process data according to outlined kernel function
        """
        inA = inputs[0]
        outA, outB = outputs

        # convert input to tensor if needed
        if inA.bcip_type != BcipEnums.TENSOR:
            inA = inA.to_tensor()

        # extract and reshape data
        if len(inA.shape) == 1:
            input_data = np.expand_dims(inA.data,axis=0)
        else:
            input_data = inA.data
        
        if outB is not None:
            outB.data = self._classifier._classifier.predict_proba(input_data)

        output_data = self._classifier._classifier.predict(input_data)
        if outA.bcip_type == BcipEnums.SCALAR:
            outA.data = int(output_data)
        else:
            outA.data = output_data
        

    @classmethod
    def add_classifier_node(cls, graph, inA, classifier, outA, outB = None, initialization_data = None, labels = None):
        """
        Factory method to create a classifier kernel and add it to a graph as a generic node object
        
        Parameters
        ----------

        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor  
            Input trial data (n_channels, n_samples)

        classifier : Classifier 
            BCIP Classifier object to be used for classification

        outA : Scalar 
            Output trial data

        outB : Tensor
            If not None, the output will be the probability of each class. Default is None

        initialization_data : Tensor 
            Initialization data to train the classifier (n_trials, n_channels, n_samples)
        
        labels : Tensor
            Labels corresponding to initialization data class labels (n_trials, )
            (n_trials, 2) for class separated data where column 1 is the trial label and column 2 is the start index
        
        """

        #create the kernel object
        c = cls(graph, inA, classifier, outA, outB, initialization_data, labels)

        params = (Parameter(inA, BcipEnums.INPUT),
                  Parameter(outA, BcipEnums.OUTPUT))
        
        if outB is not None:
            params += (Parameter(outB, BcipEnums.OUTPUT),)

        node = Node(graph, c, params)

        graph.add_node(node)

        return node
