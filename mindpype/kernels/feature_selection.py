from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor
from .kernel_utils import extract_init_inputs

from sklearn.feature_selection import SelectKBest
import numpy as np

class FeatureSelectionKernel(Kernel):
    """
    Performs feature selection using f_classif method from sklearn.feature_selection

    Parameters
    ----------

    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor
        Input data (n_samples, n_features)

    outA : Tensor
        Output data (n_samples, n_selected_features)

    initialization_data : Tensor
        Initialization data

    labels : Tensor
        Initialization data labels (n_samples, )
    """

    def __init__(self, graph, inA, outA, k=10, initialization_data=None, labels=None):
        super().__init__('FeatureSelection', MPEnums.INIT_FROM_DATA, graph)
        self.inputs = [inA]
        self.outputs = [outA]

        self._initialized = False
        self._k = k

        if initialization_data is not None:
            self.init_inputs = [initialization_data]

        if labels is not None:
            self.init_input_labels = labels


    def _initialize(self, init_inputs, init_outputs, labels):

        # check that the input init data is in the correct type
        init_in = init_inputs[0]
        accepted_inputs = (MPEnums.TENSOR,MPEnums.ARRAY,MPEnums.CIRCLE_BUFFER)

        for init_obj in (init_in,labels):
            if init_obj.mp_type not in accepted_inputs:
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

        # initialize model
        self._model = SelectKBest(k=self._k)
        self._model.fit(X, y)

        # set the initialization output
        if init_outputs[0] is not None or init_outputs[1] is not None:
            init_tensor = Tensor.create_from_data(self.session, X)

            # adjust output shapes if necessary
            if self.init_outputs[0] is not None and self.init_outputs[0].virtual:
                self.init_outputs[0].shape = (X.shape[0], self._k)

            self._process_data([init_tensor],
                               self.init_outputs)

        self._initialized = True


    def _verify(self):
        """similar verification process to individual classifier kernels"""

        # inputs must be a tensor or array of tensors
        accepted_input_types = (MPEnums.TENSOR,
                                MPEnums.ARRAY,
                                MPEnums.CIRCLE_BUFFER)

        d_in = self.inputs[0]
        if d_in.mp_type not in accepted_input_types:
            raise TypeError('Input data must be a tensor or array of tensors')

        # if input is an array, check that its elements are tensors
        if (d_in.mp_type != MPEnums.TENSOR):
            e = d_in.get_element(0)
            if e.mp_type != MPEnums.TENSOR:
                raise TypeError('Input data must be a tensor or array of tensors')

    def _process_data(self, inputs, outputs):
        """
        Process data according to outlined kernel function
        """
        inA = inputs[0]
        outA = outputs[0]

        # convert input to tensor if needed
        if inA.mp_type != MPEnums.TENSOR:
            inA = inA.to_tensor()

        # extract and reshape data
        if len(inA.shape) == 1:
            input_data = np.expand_dims(inA.data,axis=0)
        else:
            input_data = inA.data

        outA.data = self._model.transform(input_data)

    @classmethod
    def add_to_graph(cls, graph, inA, outA, k=10, init_inputs = None, labels = None):
        """
        Factory method to create a feature selection kernel and add it to a graph as a generic node object

        Parameters
        ----------

        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor
            Input data (n_channels, n_samples)

        outA : Scalar
            Output data

        init_inputs : Tensor
            Initialization data for the graph

        labels : Tensor
            Labels corresponding to initialization data class labels
        """

        # create the kernel object
        c = cls(graph, inA, outA, k, init_inputs, labels)

        params = (Parameter(inA, MPEnums.INPUT),
                  Parameter(outA, MPEnums.OUTPUT))

        node = Node(graph, c, params)

        graph.add_node(node)

        if init_inputs is not None:
            node.add_initialization_data(init_inputs, labels)

        return node
