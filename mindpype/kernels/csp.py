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
    Kernel to apply common spatial pattern (CSP) filters to trial data.

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
    inA : Tensor
        First input trial data
    outA : Tensor
        Output trial data
    init_style: MPEnum
        If `INIT_FROM_DATA` the kernel's fitlers will be computed during initialization. 
        If `INIT_FROM_COPY` the filters will be copied from a pre-existing model passed 
        in the `init_params` dictionary.
    n_filt_pairs : int, default=1
        Number of CSP filter pairs to compute. Each pair consists of eigenvectors associated 
        with the n-th largest and n-th smallest eigenvalues of the generalized eigenvalue problem.
    n_cls : int, default=2
        Number of classes in the data. This is used to determine the number of filters to compute.
    multi_class_mode : str, default='OVA'
        Mode for computing CSP filters. If 'OVA', filters are computed using a one-vs-all approach.
        If 'PW', filters are computed using a pairwise approach.
    filters : ndarray, default=None
        Pre-calculated CSP filters to be applied to input trial data. If provided, this will 
        be used to initialize the kernel.
    init_data : MindPype Tensor or Array, default=None
        Initialization data to configure the filters (n_trials, n_channels, n_samples)
    labels : MindPype Tensor or Array, default=None
        Labels corresponding to initialization data class labels (n_trials,)

    See Also
    --------
    Kernel : Base class for all kernel objects
    """

    def __init__(self,graph, inA, outA, init_style,
                 n_filt_pairs=1, n_cls=2, multi_class_mode='OVA',
                 filters=None, init_data=None, labels=None):
        """ Init """
        super().__init__('CSP',init_style,graph)
        self.inputs = [inA]
        self.outputs = [outA]

        self.n_filt_pairs = n_filt_pairs
        self.multi_class_mode = multi_class_mode
        self.n_cls = n_cls

        if init_data is not None:
            self.init_inputs = [init_data]

        if labels is not None:
            self.init_input_labels = labels

        if init_style == MPEnums.INIT_FROM_DATA:
            # model will be trained using data in tensor object at later time
            self._initialized = False
            self._W = None

        elif init_style == MPEnums.INIT_FROM_COPY:
            # model is copy of predefined MDM model object
            self._W = filters
            self._initialized = True


    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Initialize the kernel and compute intialization data output.
        Sets the filter values based on the provided initialization data.

        Parameters
        ----------
        init_inputs : list of MindPype Tensor or Array data containers
            Initialization input data container, list of length 1
        init_outputs : list of MindPype Tensor or Array data containers
            Initialization output data container, list of length 1
        labels : MindPype Tensor or Array
            Labels corresponding to initialization data class labels (n_trials,)
        """

        # check that the input init data is in the correct type
        init_in = init_inputs[0]
        accepted_inputs = (MPEnums.TENSOR, MPEnums.ARRAY, MPEnums.CIRCLE_BUFFER)

        for init_obj in (init_in,labels):
            if init_obj.mp_type not in accepted_inputs:
                raise TypeError('Initialization data must be a tensor, array, or circle buffer')

        if self.init_style == MPEnums.INIT_FROM_DATA:
            # extract the initialization data
            X = extract_init_inputs(init_in)
            y = extract_init_inputs(labels)
            self._initialized = False
            self._compute_filters(X, y)
            self._initialized = True

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
        Apply the CSP filters to the input data and
        assign the filtered data to the output tensor

        Parameters
        ----------
        inputs : list of MindPype Tensor
            Input data container, list of length 1
        outputs : list of MindPype Tensor
            Output data container, list of length 1
        """
        outputs[0].data = np.matmul(self._W.T, inputs[0].data)

    
    def _verify(self):
        """
        Verify the kernel parameters
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

        if self.n_cls < 2:
            raise ValueError('Number of classes must be greater than 1')

        if (self.n_cls > 2 and
            self.multi_class_mode not in ('OVA', 'PW')):
            raise ValueError('Invalid multi-class mode specified')

        # if the output is a virtual tensor and dimensionless,
        # add the dimensions now
        if self.n_cls == 2:
            filt_multiplier = 1
        else:
            if self.multi_class_mode == 'OVA':
                filt_multiplier = self._num_classes
            else:
                filt_multiplier = int(binom(self._num_classes,2))

        if len(d_in.shape) == 2:
            out_sz = (2*self.n_filt_pairs*filt_multiplier,d_in.shape[1])
        else:
            out_sz =  (d_in.shape[0], 2*self.n_filt_pairs*filt_multiplier, d_in.shape[2])

        if d_out.virtual and len(d_out.shape) == 0:
            d_out.shape = out_sz

        # verify that output tensor can accept data of this size
        d_out.data = np.zeros(out_sz)


    def _compute_filters(self, X, y):
        """
        Compute the CSP filters for the kernel

        Parameters
        ----------
        X : ndarray
            Initialization data to configure the filters (n_trials, n_channels, n_samples)
        y : ndarray
            Labels corresponding to initialization data class labels (n_trials,)
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

        if Nl != self.n_cls:
            raise ValueError('Number of unique labels in initialization data does not match number of classes')

        if Nl == 2:
            # binary classification, compute filters directly
            self._W = self._compute_binary_filters(X,y)

        else:
            # multi-class classification, compute filters based on mode
            _, Nc, Ns = X.shape

            if self.multi_class_mode == 'OVA':
                # one vs. all
                self._W = np.zeros((Nc,Nl*2*self.n_filt_pairs))

                for il, l in enumerate(unique_labels):
                    yl = np.copy(y)
                    yl[y==l] = 1  # target class
                    yl[y!=l] = 0  # non-target classes
                    self._W[:, il*2*self.n_filt_pairs:(il+1)*2*self.n_filt_pairs] = self._compute_binary_filters(X, yl)

            else:
                # pairwise
                Nf = int(binom(Nl, 2)) # number of pairs
                self._W = np.zeros((Nc, Nf*2*self.n_filt_pairs))

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

                    self._W[:, il*2*self.n_filt_pairs:(il+1)*2*self.n_filt_pairs] = self.compute_binary_filters(Xl, yl)


    def _compute_binary_filters(self, X, y):
        """
        Compute binary CSP filters using the generalized eigenvalue problem

        Parameters
        ----------
        X : ndarray
            Initialization data to configure the filters (n_trials, n_channels, n_samples)
        y : ndarray
            Labels corresponding to initialization data class labels (n_trials,)

        Returns
        -------
        W : ndarray
            CSP filters (n_channels, n_filters)
        """
        Nc = X.shape[1]

        # start by calculating the mean covariance matrix for each class
        C = pyriemann.utils.covariance.covariances(X)

        # remove any trials that are not positive definite
        pd = np.asarray([np.all(np.linalg.eigvals(Ci)) for Ci in C])
        C = C[pd==1]
        y = y[pd==1]

        # calculate the mean covariance matrix for each class
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

        # apply the whitening transform to the total covariance matrix
        C_tot_white = np.matmul(P,np.matmul(C_total,P.T))

        # apply the whitening transform to both class covariance matrices
        C1_bar_white = np.matmul(P,np.matmul(C_bar[0,:,:],P.T))

        # solve the generalized eigenvalue problem to get the CSP filters
        l, V = eigh(C1_bar_white, C_tot_white)

        # sort the eigenvectors in order of eigenvalues
        ix = np.flip(np.argsort(l))
        V = V[:,ix]

        # extract the specified number of filters
        W = np.concatenate((V[:,:self.n_filt_pairs], V[:,-self.n_filt_pairs:]), axis=1)

        # rotate the filters back into the channel space
        W = np.matmul(P.T,W)

        return W


    @classmethod
    def add_to_graph(cls, graph, inA, outA,
                     initialization_data=None, labels=None,
                     n_filt_pairs=2, n_cls=2, multi_class_mode='OVA',
                     filters=None):
        """
        Factory method to create a CSP filter kernel and 
        add it as a node to a graph

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to

        inA : MindPype Tensor
            Input trial data
        outA : MindPype Tensor
            Filtered trial data
        initialization_data : MindPype Tensor or Array, default=None

            Initialization data to configure the filters (n_trials, n_channels, n_samples)
        labels : MindPype Tensor or Array, default=None
            Labels corresponding to initialization data class labels (n_trials,)
        n_filt_pairs : int, default=2
            Number of CSP filter pairs to compute. Each pair consists of eigenvectors associated 
            with the n-th largest and n-th smallest eigenvalues of the generalized eigenvalue problem.
        n_cls : int, default=2
            Number of classes in the data. This is used to determine the number of filters to compute.
        multi_class_mode : str, default='OVA'
            Mode for computing CSP filters. If 'OVA', filters are computed using a one-vs-all approach.
            If 'PW', filters are computed using a pairwise approach.
        filters : ndarray, default=None
            Pre-calculated CSP filters to be applied to input trial data. If provided, this will
            be used to to define the kernel's filters and any training data will not be used to compute
            the filters.
        """

        # create the kernel object
        if filters is not None:
            k = cls(graph, inA, outA, MPEnums.INIT_FROM_COPY,
                    filters.shape[1], filters=filters)
        else:
            k = cls(graph, inA, outA, MPEnums.INIT_FROM_DATA,
                    n_filt_pairs, n_cls, multi_class_mode, 
                    init_data=initialization_data,
                    labels=labels)

        # create parameter objects for the input and output
        params = (Parameter(inA, MPEnums.INPUT),
                  Parameter(outA, MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        return node
