from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .kernel_utils import extract_init_inputs

import numpy as np
from pyriemann import classification


class RiemannMDMClassifierKernel(Kernel):
    """
    Riemannian Minimum Distance to the Mean Classifier. Kernel takes Tensor input and produces scalar label representing
    the predicted class. Review classmethods for specific input parameters

    .. note::
        This kernel utilizes the pyriemann class
        :class:`MDM <pyriemann:pyriemann.classification.MDM>`.

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor or Array
        Input data

    outA : Tensor or Scalar
        Output data

    fgda : bool
        True if the classifier should apply Fisher Geodesic Discriminant Analysis to the data

    initialization_data : Tensor
        Initialization data to train the classifier (n_trials, n_channels, n_samples)

    labels : Tensor
        Labels corresponding to initialization data class labels (n_trials, )
        (n_trials, 2) for class separated data where column 1 is the trial label and column 2 is the start index

    """

    def __init__(self,graph,inA,outA,fgda,initialization_data,labels):
        """ Init """
        super().__init__('RiemannMDM',MPEnums.INIT_FROM_DATA,graph)
        self.inputs = [inA]
        self.outputs = [outA]

        self.fgda = fgda

        self._initialized = False
        self._covariance_inputs = (0,)

        if initialization_data is not None:
            self.add_initialization_data([initialization_data], labels)

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Set the means for the classifier

        Parameters
        ----------

        init_inputs: Tensor or Array
            Input data

        init_outputs: Tensor or Scalar
            Output data
        
        labels: Tensor
            Class labels for initialization data (n_trials,)
        """
        self._train_classifier(init_inputs[0], labels)

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_in.mp_type != MPEnums.TENSOR:
            init_in = init_in.to_tensor()

        # compute init output
        if init_out is not None:
            # adjust the shape of init output tensor
            if len(init_in.shape) == 3:
                init_out.shape = (init_in.shape[0],)

            # compute the init output
            self._process_data([init_in], init_outputs)


    def _train_classifier(self, init_in, labels):
        """
        Train the classifier. The method will update the kernel's internal representation of the
        classifier

        Parameters
        ----------

        init_in: Tensor or Array
            Input data

        labels: Tensor
            Class labels for initialization data (n_trials,)
        """
        # check that the input data is valid
        if ((init_in.mp_type != MPEnums.TENSOR and
             init_in.mp_type != MPEnums.ARRAY)  or
            (labels.mp_type != MPEnums.TENSOR and
             labels.mp_type != MPEnums.ARRAY)):
                raise TypeError('RiemannianMDM kernel: invalid initialization data or labels')

        # extract the initialiation data
        X = extract_init_inputs(init_in)
        y = extract_init_inputs(labels)

        # ensure the shpaes are valid
        if len(X.shape) != 3 or len(y.shape) != 1:
            raise ValueError('RiemannianMDM kernel: invalid dimensions for initialization data or labels')

        if X.shape[0] != y.shape[0]:
            raise ValueError('RiemannianMDM kernel: number of trials in initialization data and labels must match')

        if not self.fgda:
            self.classifier = classification.MDM()
        else:
            self.classifier = classification.FgMDM()
        self.classifier.fit(X,y)


    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """

        d_in = self.inputs[0]
        d_out = self.outputs[0]

        # first ensure the input is a tensor
        if d_in.mp_type != MPEnums.TENSOR:
            raise TypeError('RiemannianMDM kernel: input must be a tensor')

        # ensure the output is a tensor or scalar
        if (d_out.mp_type != MPEnums.TENSOR and
            d_out.mp_type != MPEnums.SCALAR):
            raise TypeError('RiemannianMDM kernel: output must be a tensor or scalar')

        input_shape = d_in.shape
        input_rank = len(input_shape)

        # input tensor should not be greater than rank 3
        if input_rank > 3 or input_rank < 2:
            raise ValueError('RiemannianMDM kernel: input tensor must be rank 2 or 3')

        # if the output is a virtual tensor and dimensionless,
        # add the dimensions now
        if (d_out.mp_type == MPEnums.TENSOR and
            d_out.virtual and
            len(d_out.shape) == 0):
            if input_rank == 2:
                d_out.shape = (1,)
            else:
                d_out.shape = (input_shape[0],)

        # check for dimensional alignment
        if d_out.mp_type == MPEnums.SCALAR:
            # input tensor should only be a single trial
            if len(d_in.shape) == 3:
                # first dimension must be equal to one
                if d_in.shape[0] != 1:
                    raise ValueError('RiemannianMDM kernel: input tensor must be a single covariance matrix when using scalar output')
        else:
            # check that the dimensions of the output match the dimensions of
            # input
            if d_in.shape[0] != d_out.shape[0]:
                raise ValueError('RiemannianMDM kernel: input and output tensor must equal first dimension')

            # output tensor should be one dimensional
            if len(d_out.shape) > 1:
                raise ValueError('RiemannianMDM kernel: output tensor must be one dimensional')

    def _process_data(self, inputs, outputs):
        """
        Execute Riemann MDM classifier.

        Parameters
        ----------

        inputs: list of Tensors or Arrays
            Input data container, list of length 1
        
        outputs: list of Tensors or Scalars
            Output data container, list of length 1
        """
        input_data = inputs[0].data
        if len(inputs[0].shape) == 2:
            # pyriemann library requires input data to have 3 dimensions with the
            # first dimension being 1
            input_data = input_data[np.newaxis,:,:]

        outputs[0].data = self.classifier.predict(input_data)

    @classmethod
    def add_to_graph(cls,graph,inA,outA,fgda=False,
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
            Input data

        outA : Tensor or Scalar
            Output data

        fgda : bool
            True if the classifier should apply Fisher Geodesic Discriminant Analysis
            to the data

        initialization_data : Tensor
            Initialization data to train the classifier with (n_trials, n_channels, n_samples)

        labels : Tensor
            Class labels for initialization data (n_trials,)

        Returns
        -------
        node : Node
            Node object that contains the kernel
        """

        # create the kernel object
        k = cls(graph,inA,outA,fgda,initialization_data,labels)

        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph,k,params)

        # add the node to the graph
        graph.add_node(node)

        return node
