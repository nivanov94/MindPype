from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .kernel_utils import extract_init_inputs

import numpy as np
import pyriemann
from scipy.linalg import eigh
from scipy.special import binom
from itertools import combinations as iter_combs


class CommonSpatialPatternKernel(Kernel):
    """
    CSP Filter Kernel that applies a set of common spatial patter filters to tensors of covariance matrices

    .. note::
        This kernel utilizes the scipy functions
        :func:`eigh <scipy:scipy.linalg.eigh>`,
        :func:`binom <scipy:scipy.special.binom>`,

    .. note::
        This kernel utilizes the numpy functions
        :func:`matmul <numpy:numpy.matmul>`,
        :func:`newaxis <numpy:numpy.newaxis>`,
        :func:`unique <numpy:numpy.unique>`,
        :func:`zeros <numpy:numpy.zeros>`,
        :func:`copy <numpy:numpy.copy>`,
        :func:`concatenate <numpy:numpy.concatenate>`,
        :func:`ones <numpy:numpy.ones>`,
        :func:`asarray <numpy:numpy.asarray>`,
        :func:`all <numpy:numpy.all>`,
        :func:`mean <numpy:numpy.mean>`,
        :func:`sum <numpy:numpy.sum>`,
        :func:`isclose <numpy:numpy.isclose>`,
        :func:`diag <numpy:numpy.diag>`,
        :func:`flip <numpy:numpy.flip>`,
        :func:`argsort <numpy:numpy.argsort>`,
        :func:`eigvals <numpy:numpy.linalg.eigvals>`,
        :func:`eig <numpy:numpy.linalg.eig>`,
        :func:`squeeze <numpy:numpy.squeeze>`.

    .. note::
        This kernel utilizes the pyriemann function
        :func:`covariances <pyriemann:pyriemann.utils.covariance.covariances>`.


    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to
    inA : Tensor or Scalar
        Input data
    outA : Tensor or Scalar
        Output data

    """

    def __init__(self,graph,inA,outA,
                 init_style,init_params,
                 num_filts=2,Ncls=2,multi_class_mode='OVA'):
        """ Init """
        super().__init__('CSP',init_style,graph)
        self.inputs = [inA]
        self.outputs = [outA]

        self.num_filts = num_filts
        self._init_params = init_params
        self.multi_class_mode = multi_class_mode
        self._num_classes = Ncls

        if 'initialization_data' in init_params:
            self.init_inputs = [init_params['initialization_data']]

        if 'labels' in init_params:
            self.init_input_labels = init_params['labels']

        if init_style == MPEnums.INIT_FROM_DATA:
            # model will be trained using data in tensor object at later time
            self._initialized = False
            self._W = None

        elif init_style == MPEnums.INIT_FROM_COPY:
            # model is copy of predefined MDM model object
            self._W = init_params['filters']
            self._initialized = True


    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Set the filter values based on the provided initialization data
        """
        # check that the input init data is in the correct type
        init_in = init_inputs[0]
        accepted_inputs = (MPEnums.TENSOR,MPEnums.ARRAY,MPEnums.CIRCLE_BUFFER)

        for init_obj in (init_in,labels):
            if init_obj.mp_type not in accepted_inputs:
                raise TypeError('Initialization data must be a tensor, array, or circle buffer')

        if self.init_style == MPEnums.INIT_FROM_DATA:
            # extract the initialization data
            X = extract_init_inputs(init_in)
            y = extract_init_inputs(labels)
            self._compute_filters(X,y)

        # compute init output
        init_out = init_outputs[0]
        if init_out is not None:
            if init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

            # adjust the shape of init output tensor
            if len(init_in.shape) == 3:
                init_out.shape = (init_in.shape[0], self._W.shape[1], init_in.shape[2])

            self._process_data(init_inputs, init_outputs)


    def _process_data(self, inputs, outputs):
        """
        Process input data according to outlined kernel function

        Parameters
        ----------
        inputs : list of Tensors or Scalars
            Input data container, list of length 1
        outputs: Tensor or Scalar
            Output data container, list of length 1
        """

        outputs[0].data = np.matmul(self._W.T, inputs[0].data)

    def _compute_filters(self,X,y):
        """
        Compute CSP filters

        Parameters
        ----------
        X: Tensor
            Initialization data
        y: np.array
            Labels
        """
        # ensure the shapes are valid
        if len(X.shape) == 2:
            X = X[np.newaxis, :, :]

        if len(y.shape) == 2:
            y = np.squeeze(y)

        if len(X.shape) != 3 or len(y.shape) != 1:
            raise ValueError('Initialization data must be a 3D tensor and labels must be a 1D array')

        if X.shape[0] != y.shape[0]:
            raise ValueError('Number of trials in initialization data and labels do not match')

        unique_labels = np.unique(y)
        Nl = unique_labels.shape[0]

        if Nl != self._num_classes:
            raise ValueError('Number of unique labels in initialization data does not match number of classes')

        if Nl == 2:
            self._W = self._compute_binary_filters(X,y)

        else:
            _, Nc, Ns = X.shape

            if self.multi_class_mode == 'OVA':
                # one vs. all
                self._W = np.zeros((Nc,Nl*self.num_filts))

                for il, l in enumerate(unique_labels):
                    yl = np.copy(y)
                    yl[y==l] = 1 # target
                    yl[y!=l] = 0 # non-target
                    self._W[:, il*self.num_filts:(il+1)*self.num_filts] = self._compute_binary_filters(X,yl)

            else:
                # pairwise
                Nf = int(binom(Nl,2)) # number of pairs
                self._W = np.zeros((Nc, Nf*self.num_filts))

                for il, (l1,l2) in enumerate(iter_combs(unique_labels,2)):
                    # get trials from each label
                    Xl1 = X[y==l1,:,:]
                    Xl2 = X[y==l2,:,:]

                    # create feature and label matrices using the current label pair
                    yl = np.concatenate((l1 * np.ones(Xl1.shape[0],),
                                         l2 * np.ones(Xl2.shape[0])),
                                        axis=0)
                    Xl = np.concatenate((Xl1,Xl2),
                                        axis=0)

                    self._W[:, il*self.num_filts:(il+1)*self.num_filts] = self.compute_binary_filters(Xl, yl)

    def _compute_binary_filters(self, X, y):
        """
        Compute binary CSP filters

        Parameters
        ----------
        X: Tensor
            Initialization data
        y: np.array
            Labels

        Returns
        -------
        W: NDArray
            Binary filters which are rotated back into the channel space
        """
        _ , Nc, Ns = X.shape

        # start by calculating the mean covariance matrix for each class
        C = pyriemann.utils.covariance.covariances(X)

        # remove any trials that are not positive definite
        pd = np.asarray([np.all(np.linalg.eigvals(Ci)) for Ci in C])
        C = C[pd==1]
        y = y[pd==1]

        C_bar = np.zeros((2, Nc, Nc))
        labels = np.unique(y)
        for i, label in enumerate(labels):
            C_bar[i,:,:] = np.mean(C[y==label,:,:], axis=0)

        C_total = np.sum(C_bar, axis = 0)

        # get the whitening matrix
        d, U = np.linalg.eig(C_total)

        # filter any eigenvalues close to zero
        d[np.isclose(d, 0)] = 0
        U = U[:,d!=0]
        d = d[d!=0]

        # construct the whitening matrix
        P = np.matmul(np.diag(d ** (-1/2)), U.T)

        C_tot_white = np.matmul(P,np.matmul(C_total,P.T))

         # apply the whitening transform to both class covariance matrices
        C1_bar_white = np.matmul(P,np.matmul(C_bar[0,:,:],P.T))

        l, V = eigh(C1_bar_white, C_tot_white)

        # sort the eigenvectors in order of eigenvalues
        ix = np.flip(np.argsort(l))
        V = V[:,ix]

        # extract the specified number of filters
        m = self.num_filts // 2
        W = np.concatenate((V[:,:m], V[:,-m:]), axis=1)

        # rotate the filters back into the channel space
        W = np.matmul(P.T,W)

        return W


    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """

        # first ensure the input and output are tensors
        d_in = self.inputs[0]
        d_out = self.outputs[0]

        for param in (d_in, d_out):
            if param.mp_type != MPEnums.TENSOR:
                raise TypeError('Input and output parameters must be tensors')

        # input tensor should be two- or three-dimensional
        if len(d_in.shape) != 2 and len(d_in.shape) != 3:
            raise ValueError('Input tensor must be two- or three-dimensional')

        if self._num_classes < 2:
            raise ValueError('Number of classes must be greater than 1')

        if (self._num_classes > 2 and
            self.multi_class_mode not in ('OVA', 'PW')):
            raise ValueError('Invalid multi-class mode specified')

        # if the output is a virtual tensor and dimensionless,
        # add the dimensions now
        if self._num_classes == 2:
            filt_multiplier = 1
        else:
            if self.multi_class_mode == 'OVA':
                filt_multiplier = self._num_classes
            else:
                filt_multiplier = int(binom(self._num_classes,2))

        if len(d_in.shape) == 2:
            out_sz = (self.num_filts*filt_multiplier,d_in.shape[1])
        else:
            out_sz =  (d_in.shape[0], self.num_filts*filt_multiplier, d_in.shape[2])

        if d_out.virtual and len(d_out.shape) == 0:
            d_out.shape = out_sz

        # verify that output tensor can accept data of this size
        d_out.data = np.zeros(out_sz)

    @classmethod
    def add_to_graph(cls,graph,inA,outA,
                     initialization_data=None,labels=None,
                     num_filts=2,Ncls=2,multi_class_mode='OVA',
                     filters=None):
        """
        Factory method to create a CSP filter node and add it to a graph

        Note that the node will have to be initialized prior
        to execution of the kernel.

        Parameters
        ----------

        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor or Scalar
            Input data

        outA : Tensor or Scalar
            Output data

        initialization_data : Tensor
            Initialization data to configure the filters (n_trials, n_channels, n_samples)

        labels : Tensor
            Labels corresponding to initialization data class labels (n_trials, )

        num_filts : int
            Number of spatial filters to apply to trial data.

        filters : ndarray
            Pre-calculated spatial filters to be applied to input trial data.
            If provided, this will be used to initialize the kernel.

        """

        # create the kernel object
        if filters:
            init_params = {'filters' : filters}
            k = cls(graph,inA,outA,MPEnums.INIT_FROM_COPY,init_params,filters.shape[1])
        else:
            init_params = {'initialization_data' : initialization_data,
                           'labels'              : labels}

            k = cls(graph,inA,outA,MPEnums.INIT_FROM_DATA,init_params,
                    num_filts,Ncls,multi_class_mode)

        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph,k,params)

        # add the node to the graph
        graph.add_node(node)

        return node
