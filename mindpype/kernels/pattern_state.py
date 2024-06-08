from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

import numpy as np
import pyriemann
import sklearn


class PatternStateKernel(Kernel):
    """
    Kernel to predict the pattern state of a EEG signal.
    The states are defined by applying a clustering
    algorithm to the EEG signal.

    Parameters
    ----------
    graph : Graph
        Graph object to which the kernel is associated.
    inA : Tensor
        Input data
    outA : Tensor or Scalar
        Predicted pattern state
    outB : Tensor
        Pattern state probabilities
    k_selection_method : str, default='pred_strength'
        Method to select the number of clusters. Either 
        'pred_strength' or 'fixed'. If 'pred_strength' is
        selected, the number of clusters is selected based
        on the prediction strength of clustering method.
        If 'fixed' is selected, the number of clusters is
        fixed to the value specified in the parameter n_clusters.
    n_clusters : int, default=2
        Number of clusters to use for clustering. Only used
        if k_selection_method is set to 'fixed'.
    k_sel_thresh : float, default=0.5
        Threshold used for the prediction strength of clustering
        algorithm. Only used if k_selection_method is set to
        'pred_strength'.
    k_range : tuple, default=(2, 10)
        Range of cluster numbers to consider when selecting the
        number of clusters using the prediction strength method.
        Only used if k_selection_method is set to 'pred_strength'.
    tangent_space : bool, default=True
        Whether to apply a tangent space transformation to the 
        input covariance data prior to clustering.
    scale_data : bool, default=True
        Whether to scale the input data prior to clustering.
    update_rate : float, default=0.1
        Learning rate for updating the clustering model.
    init_data : Tensor, default=None
        Initial data to use for the kernel. If None, the kernel
        will use initialization data produced by a previous node
        in the graph.
    init_labels : Tensor, default=None
        Initial labels to use for the kernel. If None, the kernel
        will use initialization labels produced by a previous node
        in the graph.
    
    See Also
    --------
    Kernel: Base class for all kernels.
    """

    def __init__(self, graph, inA, outA, outB, 
                 k_selection_method='pred_strength', 
                 n_clusters=2, k_sel_thresh=0.5, 
                 tangent_space=True, scale_data=True,
                 update_rate=0.1,
                 init_data=None, init_labels=None):
        """Init."""
        super().__init__('PatternState', MPEnums.INIT_FROM_DATA, graph)
        self.inputs = [inA]
        self.outputs = [outA, outB]

        # mark the input as being a covariance input
        self._covariance_inputs = (0,)

        self._k_selection_method = k_selection_method
        self._n_clusters = n_clusters
        self._k_sel_thresh = k_sel_thresh
        self._tangent_space = tangent_space

        if init_data is not None:
            self.init_inputs = [init_data]
        
        if init_labels is not None:
            self.init_input_labels = [init_labels]

        self._initialized = False
        self._mod = None

    
    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Initialize the kernel and perform clustering.

        Parameters
        ----------
        init_inputs : list of Tensors
            List of input data tensors. Should contain a single tensor.
        init_outputs : list of Tensors
            List of output data tensors.
        labels : list of Tensors
            Not used. Here for compatibility with the Kernel class.
        """

        # mark the kernel as uninitialized
        self._initialized = False

        # check if the input data is valid
        init_in = init_inputs[0]
        if init_in is None or init_in.mp_type != MPEnums.TENSOR:
            raise TypeError('PatternStateKernel: input data must be a Tensor')
        
        # extract the training data
        X = init_in.data

        pipeline = []
        if self._tangent_space:
            pipeline.append(('tangent', pyriemann.tangentspace.TangentSpace()))

            if self._scale_data:
                pipeline.append(('scaler', sklearn.preprocessing.StandardScaler()))

        if self._k_selection_method == 'pred_strength':
            if len(pipeline) > 0:
                # create a pipeline with the preprocessing steps and apply it
                pre_pipe = sklearn.pipeline.Pipeline(pipeline)
                X = pre_pipe.fit_transform(X)
            
            # determine the number of clusters using the prediction strength
            # of the clustering algorithm
            self._n_clusters = self._prediction_strength(X)
        
        # add the clustering step to the pipeline
        if self._tangent_space:
            pipeline.append(('clustering', sklearn.cluster.KMeans(n_clusters=self._n_clusters)))
        else:
            pipeline.append(('clustering', pyriemann.clustering.Kmeans(n_clusters=self._n_clusters)))
        
        # create the final pipeline and fit it to the data
        self._mod = sklearn.pipeline.Pipeline(pipeline)
        self._mod.fit(X)

        # mark the kernel as initialized
        self._initialized = True

        # compute the init outputs
        if init_outputs[0] is not None or init_outputs[1] is not None:
            # adjust the output shapes if necessary
            if init_outputs[0] is not None and init_outputs[0].virtual:
                init_outputs[0].shape = (X.shape[0],)

            if init_outputs[1] is not None and init_outputs[1].virtual:
                init_outputs[1].shape = (X.shape[0], self._n_clusters)

            # compute the output data
            self._process_data(init_inputs, init_outputs)

    
    def _update(self, update_inputs, update_outputs, labels):
        """
        Update the kernel using new data.

        Parameters
        ----------
        update_inputs : list of Tensors
            List of input data tensors, list of length 1
        update_outputs : list of Tensors
            List of output data tensors, list of length 2
        labels : list of Tensors
            Not used. Here for compatibility with the Kernel class.
        """
        if not self._initialized:
            # not initialized, so initialize the kernel instead
            self._initialize(update_inputs, update_outputs, labels)
            return

        # extract the update data
        X = update_inputs[0].data

        # update the clustering model
        self._mod.partial_fit(X, sample_weight=self._update_rate)

        # compute the output data
        self._process_data(update_inputs, update_outputs)


    def _process_data(self, inputs, outputs):
        """
        Predict the pattern state of the input data.

        Parameters
        ----------
        inputs : list of Tensors
            List of input data tensors, list of length 1
        outputs : list of Tensors
            List of output data tensors, list of length 2
        """
        inA = inputs[0]

        if len(inA.shape) == 2:
            local_input_data = np.expand_dims(inA.data, 0)
        else:
            local_input_data = inA.data

        if outputs[0] is not None:
            outputs[0].data = self._mod.predict(local_input_data)

        if outputs[1] is not None:
            outputs[1].data = self._mod.predict_proba(local_input_data)


    def _prediction_strength(self, X):
        """
        Perform the prediction strength of the clustering algorithm.
        """
        # determine the range of cluster numbers to consider
        k_range = range(self._k_range[0], self._k_range[1])
        ps = np.ones((len(k_range), 10))

        for i_r in range(10):

            # split the data into training and testing sets
            Xtr, Xte = sklearn.model_selection.train_test_split(X, 
                                                                test_size=0.5,
                                                                shuffle=True,
                                                                random_state=i_r)

            # iterate over each cluster number
            for i_k, k in enumerate(k_range):
                # initialize the clustering models
                tr_mod = sklearn.cluster.KMeans(n_clusters=k)
                te_mod = sklearn.cluster.KMeans(n_clusters=k)

                # fit the models to the training data
                tr_mod.fit(Xtr)
                te_mod.fit(Xte)

                # compute the prediction strength of the clustering algorithm
                ps[i_k, i_r] = _compute_pred_strength(k, tr_mod, te_mod, Xte)

        ps_score = np.mean(ps, axis=1) + np.std(ps, axis=1)/np.sqrt(10)
        ps_score[ps_score < self._k_sel_thresh] = np.inf # filter out values below the threahold
        return k_range[np.argmin(ps_score)]


    @classmethod
    def add_to_graph(cls, graph, inA, outA=None, outB=None,
                     k_selection_method='pred_strength', 
                     n_clusters=2, k_sel_thresh=0.5, 
                     k_range=(2, 10), tangent_space=True,
                     update_rate=0.1,
                     scale_data=True, init_data=None, 
                     init_labels=None):
        """
        Factory method to create a PatternStateKernel and add it to the graph.

        Parameters
        ----------
        graph : Graph
            Graph object to which the kernel is associated.
        inA : Tensor
            Input data
        outA : Tensor or Scalar, default=None
            Predicted pattern state
        outB : Tensor, default=None
            Pattern state probabilities
        k_selection_method : str, default='pred_strength'
            Method to select the number of clusters. Either 
            'pred_strength' or 'fixed'. If 'pred_strength' is
            selected, the number of clusters is selected based
            on the prediction strength of clustering method.
            If 'fixed' is selected, the number of clusters is
            fixed to the value specified in the parameter n_clusters.
        n_clusters : int, default=2
            Number of clusters to use for clustering. Only used
            if k_selection_method is set to 'fixed'.
        k_sel_thresh : float, default=0.5
            Threshold used for the prediction strength of clustering
            algorithm. Only used if k_selection_method is set to
            'pred_strength'.
        k_range : tuple, default=(2, 10)
            Range of cluster numbers to consider when selecting the
            number of clusters using the prediction strength method.
            Only used if k_selection_method is set to 'pred_strength'.
        tangent_space : bool, default=True
            Whether to apply a tangent space transformation to the 
            input covariance data prior to clustering.
        scale_data : bool, default=True
            Whether to scale the input data prior to clustering.
        update_rate : float, default=0.1
            Learning rate for updating the clustering model.
        init_data : Tensor, default=None
            Initial data to use for the kernel. If None, the kernel
            will use initialization data produced by a previous node
            in the graph.
        init_labels : Tensor, default=None
            Initial labels to use for the kernel. If None, the kernel
            will use initialization labels produced by a previous node
            in the graph.

        Returns
        -------
        PatternStateKernel
            The created PatternStateKernel object.
        """
        # create the kernel object
        k = cls(graph, inA, outA, outB, k_selection_method,
                n_clusters, k_sel_thresh,
                k_range, tangent_space, scale_data,
                update_rate,
                init_data, init_labels)
        
        # create parameters for the kernel
        params = (Parameter(inA, MPEnums.INPUT),)

        if outA is not None:
            params += (Parameter(outA, MPEnums.OUTPUT),)

        if outB is not None:
            params += (Parameter(outB, MPEnums.OUTPUT),)

        # create a node for the kernel
        n = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(n)

        # add initialization data if necessary
        if init_data is not None:
            n.add_initialization_data(init_data, init_labels)

        return n


# k selection
def _compute_pred_strength(k, tr_mod, te_mod, Xte):
    """
    Computes the prediction strength of a clustering metric
    for a given clustering model and dataset.

    Parameters
    ----------
    k : int
        Number of clusters
    tr_mod : sklearn model
        Clustering model initialized using the training set
    te_mod : sklearn model
        Clustering model initialized using the test set
    Xte : array-like
        Test set data

    Returns
    -------
    ps : float
        Prediction strength of the clustering metric
    """
    # get the predictions of both clustering models
    yte = te_mod.predict(Xte)
    ytr = tr_mod.predict(Xte)

    # get the indices of the samples in each cluster
    # from the test model
    Ak = np.stack([yte==ki for ki in range(k)])
    nk = np.sum(Ak, axis=1)
    
    # initialize the metric
    ps = np.inf

    # iterate over each cluster
    for i_k in range(k):
        # extract the cluster labels from the training set model
        # for all of the samples in the i_kth cluster within the
        # testing set model
        ytr_i = ytr[Ak[i_k]]

        # compute the number of sample pairs within the
        # i_kth test set cluster that are in the same 
        # training set cluster
        s = 0
        for i, yi in enumerate(ytr_i):
            for j, yj in enumerate(ytr_i):
                if i != j and yi == yj:
                    s += 1

        # compute the prediction strength metric for the i_kth cluster
        if s != 0:
            s = s / (nk[i_k] * (nk[i_k]-1))

        # update the overall prediction strength metric
        if s < ps:
            ps = s

    return ps


class MultiPotatoClustering(sklearn.base.BaseEstimator, 
                            sklearn.base.ClassifierMixin,
                            sklearn.base.ClusterMixin,
                            sklearn.base.TransformerMixin):
    
    """
    Defines an extension to the sklearn KMeans clustering algorithm
    that uses the KMeans clusters to define a set of K Riemannian 
    Potato models. The Potato models are used to classify the data
    into K+1 different classes. The (K+1)th class is the outlier class.


    """

    def __init__(self, n_clusters=2, threshold=2.0):
        """Init."""
        self.n_clusters = n_clusters
        self.threshold = threshold
        self._state_models = []


    def fit(self, X, y=None):
        """
        Fit the clustering model to the data.

        Parameters
        ----------
        X : array-like
            Input data
        y : None
            Not used. Here for compatibility with sklearn API.        
        """

        # first use KMeans to define the initial set of clusters
        km = sklearn.cluster.KMeans(n_clusters=self.n_clusters)
        km.fit(X)

        # create a set of Riemannian Potato models
        for i in range(self.n_clusters):
            # extract the data for the i-th cluster
            X_i = X[km.labels_ == i]

            # create a Potato model for the i-th cluster
            potato = pyriemann.classification.Potato()
            potato.fit(X_i)

            self._state_models.append(potato)

        return self
    

    def predict(self, X):
        """
        Predict the class of the input data.

        Parameters
        ----------
        X : array-like
            Input data

        Returns
        -------
        y : array-like
            Predicted class labels
        """
        dists = self.transform(X)

        # check if any samples should be classified to
        # the outlier class
        min_dists = np.min(dists, axis=1)
        y = np.argmin(dists, axis=1)
        y[min_dists > self.threshold] = self.n_clusters

        return y
    

    def transform(self, X):
        """
        Transform the input data using the Riemannian Potato models.

        Parameters
        ----------
        X : array-like
            Input data

        Returns
        -------
        dists : array-like
            Distance of the input data to each of the potato models
        """
        dists = np.zeros(X.shape[0], self.n_clusters)

        # compute the distance of the input data to each of the 
        # potato models
        for i, potato in enumerate(self._state_models):
            dists[:,i] = potato.transform(X)

        return dists
    

    def update(self, X, alpha=0.1):
        """
        Update the Riemannian Potato models using new data.

        Parameters
        ----------
        X : array-like
            New data to use for updating the models

        alpha : float, default=0.1
            Learning rate for the update
        
        Returns
        -------
        self : MultiPotatoClustering
            The updated model
        """
        # predict the state for each new sample
        y = self.predict(X)

        # update the potato models
        for i, potato in enumerate(self._state_models):
            X_i = X[y == i]
            potato.partial_fit(X_i, alpha)

        return self



