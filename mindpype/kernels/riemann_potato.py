from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor
from .kernel_utils import extract_init_inputs

import numpy as np

from pyriemann.clustering import Potato
from pyriemann.utils.covariance import covariances
import numpy as np


class RiemannPotatoKernel(Kernel):
    """
    Riemannian potato artifact detection detector.
    Kernel takes Tensor input and produces scalar label representing
    the predicted class

    Parameters
    ----------

    graph : Graph
        Graph that the kernel should be added to

    inputA : Tensor or Array
        Input data

    outputA : Tensor or Scalar
        Output data

    out_score :


    """

    def __init__(self,graph,inA,outA,thresh,max_iter,regulization,
                 initialization_data=None):
        """ Init """
        super().__init__('RiemannPotato',MPEnums.INIT_FROM_DATA,graph)
        self.inputs = [inA]
        self.outputs = [outA]

        self._thresh = thresh
        self._max_iter = max_iter
        self._r = regulization

        if initialization_data is not None:
            self.init_inputs = [initialization_data]

        # model will be trained using data in tensor object at later time
        self._initialized = False
        self._potato_filter = None

        self._covariance_inputs = (0,)



    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Set reference covariance matrix, mean, and standard deviation

        Parameters
        ----------

        init_inputs: Tensor or Array
            Input data

        init_outputs: Tensor or Scalar
            Output data

        labels: None
        """

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        self._fit_filter(init_in)

        # compute init output
        if init_out is not None and init_in is not None:
            # adjust the shape of init output tensor
            if init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()
            if len(init_in.shape) == 3:
                init_out.shape = (init_in.shape[0],)

            # if the init input is trial data, compute the covariances
            if init_in.shape[-2] != init_in.shape[-1]:
                init_trial_data = init_in.data
                init_covs = covariances(init_trial_data)
                init_in = Tensor.create_from_data(self.session, init_covs)

            self._process_data([init_in], init_outputs)

    def _fit_filter(self, init_in):
        """
        Fit the potato filter using the initialization data

        Parameters
        ----------

        init_in: Tensor or Array
            Input initialization data
        """
        # check that the input data is valid
        if (init_in.mp_type != MPEnums.TENSOR and
            init_in.mp_type != MPEnums.ARRAY  and
            init_in.mp_type != MPEnums.CIRCLE_BUFFER):
            raise TypeError("Riemannian potato kernel: Initialization data must be a Tensor or Array")

        # extract the initialization data
        X = extract_init_inputs(init_in)

        if len(X.shape) != 3:
            raise ValueError("Riemannian potato kernel: Initialization data must be a 3D Tensor")

        if X.shape[-2] != X.shape[-1]:
            # convert to covs
            X = covariances(X)
            X = (1-self._r)*X + self._r*np.eye(X.shape[-1])

        self._potato_filter = Potato(threshold=self._thresh, n_iter_max=self._max_iter)
        self._potato_filter.fit(X)

    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        d_in = self.inputs[0]
        d_out = self.outputs[0]

        # first ensure the input is a tensor
        if d_in.mp_type != MPEnums.TENSOR:
            raise TypeError("Riemannian potato kernel: Input data must be a Tensor")

        # ensure the output is a tensor or scalar
        if (d_out.mp_type != MPEnums.TENSOR and
            d_out.mp_type != MPEnums.SCALAR):
            raise TypeError("Riemannian potato kernel: Output data must be a Tensor or Scalar")

        # check thresh and max iterations
        if self._thresh < 0:
            raise ValueError("Riemannian potato kernel: Threshold must be greater than 0")

        if self._max_iter < 0:
            raise ValueError("Riemannian potato kernel: Maximum iterations must be greater than 0")

        # check in/out dimensions
        input_shape = d_in.shape
        input_rank = len(input_shape)

        if input_rank > 3 or input_rank < 2:
            raise ValueError("Riemannian potato kernel: Input tensor must be rank 2 or 3")

        # input should be a covariance matrix
        if input_shape[-2] != input_shape[-1]:
            raise ValueError("Riemannian potato kernel: Input tensor must be a covariance matrix")

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
                    raise ValueError("Riemannian potato kernel: Input tensor must be a single covariance matrix when using scalar output")
        else:
            # check that the dimensions of the output match the dimensions of
            # input
            if d_in.shape[0] != d_out.shape[0]:
                raise ValueError("Riemannian potato kernel: Input and output tensor must have equal first dimension")

            # output tensor should be one dimensional
            if len(np.squeeze(d_out.shape)) > 1:
                raise ValueError("Riemannian potato kernel: Output tensor must be one dimensional")

    def _process_data(self, inputs, outputs):
        """
        TODO: description

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


        input_data = (1-self._r)*input_data + self._r*np.eye(inputs[0].shape[-1])
        outputs[0].data = self._potato_filter.predict(input_data)

    @classmethod
    def add_to_graph(cls,graph,inA,outA, initialization_data=None,
                     thresh=3,max_iter=100,regularization=0.01):
        """
        Factory method to create a riemann potato artifact detector

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor or Array
            Input data

        outA : Tensor or Scalar
            Output data

        initialization_data : Tensor or Array
            Data used to initialize the model

        thresh : float, default = 3
            Threshold for the potato filter

        max_iter : int, default = 100
            Maximum number of iterations for the potato filter

        regularization : float, default = 0.01
            Regularization parameter for the potato filter
        """

        # create the kernel object

        k = cls(graph,inA,outA,thresh,max_iter,regularization,
                initialization_data)

        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA, MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph,k,params)

        # add the node to the graph
        graph.add_node(node)

        return node
