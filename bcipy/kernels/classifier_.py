import imp
from types import NoneType

from matplotlib.cm import ScalarMappable
from classes.kernel import Kernel
from classes.node import Node 
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.array import Array
from classes.classifier import Classifier
from classes.bcip_enums import BcipEnums

from sklearn import * 
import numpy as np

class ClassifierKernel(Kernel):
    """
    Classify
    """

    def __init__(self, graph, inputA, classifier, outputA, initialization_data, labels, conf = None, pred_proba = None):
        super().__init__('Classifier', BcipEnums.INIT_FROM_DATA, graph)
        self._inputA = inputA
        self._classifier = classifier
        self._outputA = outputA
        self._initialization_data = initialization_data 
        self._labels = labels


        self._initialized = False
        self._init_inA = None
        self._init_outA = None

        self._graph = graph

        if self._classifier._ctype == 'lda':
            self._conf = conf
            self._pred_proba = pred_proba


    def initialize(self):

        if self._initialization_data == None:
            self._initialization_data = self._init_inA

        print(f"init_in ID: {self._init_inA._id}")

        sts1,sts2 = BcipEnums.SUCCESS, BcipEnums.SUCCESS

        sts1 = self.train_classifier()
        if self._init_outA.__class__ != NoneType:
            sts2 = self.initialization_execution()
        
        if sts1 != BcipEnums.SUCCESS:
            return sts1
        elif sts2 != BcipEnums.SUCCESS:
            return sts2
        else:
            return BcipEnums.SUCCESS
        
    def train_classifier(self):
        if self._initialization_data == None:
            self._initialization_data = self._init_inA
        
        if (not (isinstance(self._initialization_data,Tensor) or 
                 isinstance(self._initialization_data, Array))) or \
                (not (isinstance(self._labels,Tensor) or 
                 isinstance(self._labels, Array))):
                return BcipEnums.INITIALIZATION_FAILURE
        
        # ensure the shpaes are valid
        if len(self._initialization_data.shape) == 3:
            temp_tensor = self._initialization_data.data
            index1, index2, index3 = self._initialization_data.shape
            temp_tensor = np.reshape(temp_tensor, (index1, index2 * index3))

        if (len(temp_tensor.shape) != 2 or len(self._labels.shape) != 1):
            return BcipEnums.INITIALIZATION_FAILURE
        
        if temp_tensor.shape[0] != self._labels.shape[0]:
            return BcipEnums.INITIALIZATION_FAILURE

        #print(self._initialization_data.shape, self._labels.shape)
        #print(self._initialization_data.data)
        #all_zeros = not np.any(self._initialization_data.data)
        #print(all_zeros)
        try:
            print(np.shape(temp_tensor))
            self._classifier._classifier.fit(temp_tensor, self._labels.data)
        except:
            return BcipEnums.INITIALIZATION_FAILURE

        self._initialized = True
        return BcipEnums.SUCCESS


    def verify(self):
        """similar verification process to individual classifier kernels"""

        if self._initialization_data == None:
            self._graph._missing_data = True

        if (not isinstance(self._inputA, Tensor)) or \
           (not isinstance(self._outputA, Scalar)) or \
           (not isinstance(self._classifier, Classifier)):
            return BcipEnums.INVALID_PARAMETERS
        input_shape = self._inputA.shape
        input_rank = len(input_shape)

        if input_rank == 0:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS
        

    def execute(self):
        """execute the kernel function using the scipy predict function"""
        
        if len(self._inputA.shape) == 2:

            temp_input = np.reshape(self._inputA.data, (1, self._inputA.shape[0]*self._inputA.shape[1]))
            temp_tensor = Tensor.create_from_data(self._session, np.shape(temp_input), temp_input)
            print(temp_tensor.shape)
        
        return self.process_data(temp_tensor, self._outputA)


    def initialization_execution(self):

        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data, output_data):
        # fix
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        

        out = self._classifier._classifier.predict(input_data.data)

        

        if isinstance(output_data,Scalar):
            #output_data.data = int(out)
            output_data.data = int(out)
        else:
            output_data.data = out
        
        return BcipEnums.SUCCESS


    @classmethod
    def add_classifier_node(cls, graph, inputA, classifier, outputA, initialization_data, labels):
        """Factory method to create a classifier kernel and add it to a graph as a generic node object"""

        #create the kernel object
        c = cls(graph, inputA, classifier, outputA, initialization_data, labels)

        params = (Parameter(inputA, BcipEnums.INPUT),
                  Parameter(outputA, BcipEnums.OUTPUT))

        node = Node(graph, c, params)

        graph.add_node(node)

        return node