from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor
from .kernel_utils import extract_nested_data


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


    def initialize(self):

        sts = BcipEnums.SUCCESS

        self._initialized = False # clear initialized flag
        
        # check that the input init data is in the correct type
        init_in = self.init_inputs[0]
        labels = self.init_input_labels
        accepted_inputs = (BcipEnums.TENSOR,BcipEnums.ARRAY,BcipEnums.CIRCLE_BUFFER)
        
        for init_obj in (init_in,labels):
            if init_obj.bcip_type not in accepted_inputs:
                return BcipEnums.INITIALIZATION_FAILURE
    
        # extract the initialization data from a potentially nested array of tensors 
        if init_in.bcip_type == BcipEnums.TENSOR: 
            X = init_in.data
        else:
            try:
                # extract the data from a potentially nested array of tensors
                X = extract_nested_data(init_in)
            except:
                return BcipEnums.INITIALIZATION_FAILURE    
    
        if labels.bcip_type == BcipEnums.TENSOR:    
            y = labels.data
        else:
            try:
                y = extract_nested_data(labels)
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

        # initialize the classifier
        try:
            self._classifier._classifier.fit(X, y)
        except:
            return BcipEnums.INITIALIZATION_FAILURE

        self._initialized = True

        # set the initialization output        
        if (sts == BcipEnums.SUCCESS and 
            (self.init_outputs[0] is not None or self.init_outputs[1] is not None)):
            init_tensor = Tensor.create_from_data(self.session, X.shape, X)

            # adjust output shapes if necessary
            if self.init_outputs[0] is not None and self.init_outputs[0].virtual:
                self.init_outputs[0].shape = (X.shape[0],)
            
            if self.init_outputs[1] is not None and self.init_outputs[1].virtual:
                self.init_outputs[1].shape = (X.shape[0], self._classifier._classifier.n_classes_)

            sts = self._process_data(init_tensor, 
                                     self.init_outputs[0], 
                                     self.init_outputs[1])

            # pass on the labels
            self.copy_init_labels_to_output()
        
        return sts
        

    def verify(self):
        """similar verification process to individual classifier kernels"""

        # inputs must be a tensor or array of tensors
        accepted_input_types = (BcipEnums.TENSOR, 
                                BcipEnums.ARRAY, 
                                BcipEnums.CIRCLE_BUFFER)
        
        d_in = self.inputs[0]
        if d_in._bcip_type not in accepted_input_types:
            return BcipEnums.INVALID_PARAMETERS

        # if input is an array, check that its elements are tensors
        if (d_in.bcip_type != BcipEnums.TENSOR):
            e = d_in.get_element(0)
            if e.bcip_type != BcipEnums.TENSOR:
                return BcipEnums.INVALID_PARAMETERS

        # check that the classifier is valid
        if (self._classifier.bcip_type != BcipEnums.CLASSIFIER):
            return BcipEnums.INVALID_PARAMETERS
        
        # ensure the classifier has a predict method
        if (not hasattr(self._classifier._classifier, 'predict') or 
            not callable(self._classifier._classifier.predict)):
            return BcipEnums.INVALID_PARAMETERS
        
        # if using probability output, ensure the classifier has a predict_proba method
        if (self.outputs[1] is not None and
            (not hasattr(self._classifier._classifier, 'predict_proba') or
             not callable(self._classifier._classifier.predict_proba))):
            return BcipEnums.INVALID_PARAMETERS

        # verify type and shape of outputs
        if d_in.bcip_type == BcipEnums.TENSOR:
            input_sz = d_in.shape
        else:
            input_sz = (d_in.capacity,) + d_in.get_element(0).shape


        for i_o, d_out in enumerate(self.outputs):
            if d_out is not None: # skip optional outputs not used
                # output must be scalar or tensor
                accepted_output_types = (BcipEnums.SCALAR, BcipEnums.TENSOR)
                if d_out.bcip_type not in accepted_output_types:
                    return BcipEnums.INVALID_PARAMETERS
        
                # verify input and output dimensions
                if d_in.bcip_type == BcipEnums.TENSOR:
                    if len(input_sz) == 1:
                        # single trial/sample mode
                        if d_out.bcip_type == BcipEnums.TENSOR:
                            if i_o == 0:
                                # predicted label output
                                output_sz = (1,)
                            else:
                                # probability output
                                output_sz = (1, self._classifier._classifier.n_classes_)
                
                    elif len(input_sz) == 2:
                        #single trial or multi-trial batch mode
                        if d_out.bcip_type == BcipEnums.TENSOR:
                            if i_o == 0:
                                # predicted label output
                                output_sz = (input_sz[0],)
                            else:
                                # probability output
                                output_sz = (input_sz[0], self._classifier._classifier.n_classes_)

                    elif (d_out.bcip_type == BcipEnums.SCALAR and
                          input_sz[0] != 1):
                        return BcipEnums.INVALID_PARAMETERS
                else:
                    # input is an array
                    if (d_out.bcip_type == BcipEnums.SCALAR):
                        return BcipEnums.INVALID_PARAMETERS

                    # check elements are correct shape
                    if len(input_sz) == 2:
                        if i_o == 0:
                            # predicted label output
                            output_sz = (d_in.capacity,)
                        else:
                            # probability output
                            output_sz = (d_in.capacity, self._classifier._classifier.n_classes_)
                    else:
                        return BcipEnums.INVALID_PARAMETERS

                if d_out.bcip_type == BcipEnums.TENSOR:
                    if d_out.virtual and len(d_out.shape) == 0:
                        d_out.shape = output_sz

                    if d_out.shape != output_sz:
                        return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS
        

    def execute(self):
        """
        Execute single trial classification
        """
        # if input is not a tensor, convert
        if self.inputs[0].bcip_type != BcipEnums.TENSOR:
            input_tensor = self.inputs[0].to_tensor()
        else:
            input_tensor = self.inputs[0]

        return self._process_data(input_tensor, self.outputs[0], self.outputs[1])


    def _process_data(self, inA, outA, outB=None):
        """
        Process data according to outlined kernel function
        """
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        
        if len(inA.shape) == 1:
            input_data = np.expand_dims(inA.data,axis=0)
        else:
            input_data = inA.data
        
        if outB is not None:
            output_prob = self._classifier._classifier.predict_proba(input_data)
            if outB.bcip_type == BcipEnums.SCALAR:
                outB.data = float(output_prob)
            else:
                outB.data = output_prob

        output_data = self._classifier._classifier.predict(input_data)

        if outA.bcip_type == BcipEnums.SCALAR:
            outA.data = int(output_data)
        else:
            outA.data = output_data
        
        return BcipEnums.SUCCESS


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
