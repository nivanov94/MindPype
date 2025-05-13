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
    outlier_state : bool, default=False
        Whether the model should include an outlier state. If True,
        the predictions will be made using a set of Riemannian Potato
        models. If the input data is not close to any of the potato
        models, it will be classified as an outlier.
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

    def __init__(
            self, graph, inA, outA, outB, 
            k_selection_method='pred_strength', 
            n_clusters=2, k_sel_thresh=0.5, k_range=(2, 10),
            tangent_space=True, scale_data=True,
            outlier_state=False, update_rate=0.1,
            init_data=None, init_labels=None
        ):
        """Init."""
        super().__init__('PatternState', MPEnums.INIT_FROM_DATA, graph)
        self.inputs = [inA]
        self.outputs = [outA, outB]

        # determine the mode of the kernel
        if outlier_state:
            self._mode = 'potato'
        elif tangent_space:
            self._mode = 'tangent_space'
        else:
            self._mode = 'covariance'

        # mark the input as being a covariance input
        self._covariance_inputs = (0,)

        self._k_selection_method = k_selection_method
        self._n_clusters = n_clusters
        self._k_sel_thresh = k_sel_thresh
        self._tangent_space = tangent_space
        self._scale_data = scale_data
        self._update_rate = update_rate

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
                Xk = pre_pipe.fit_transform(X)
            
            # determine the number of clusters using the prediction strength
            # of the clustering algorithm
            self._n_clusters = self._prediction_strength(Xk)
        
        # add the clustering step to the pipeline
        if self._mode == 'potato':
            pipeline = [('clustering', MultiPotatoClustering(n_clusters=self._n_clusters))]
        elif self._mode == 'tanget_space':
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
        if self._mode == 'potato':
            self._mod.update(X, alpha=self._update_rate)
        else:
            # identify the state of the new samples
            y = self._mod.predict(X)

            for i in range(self._n_clusters):
                clust_mod = self._mod.named_steps['clustering']
                # compute the means within each cluster with
                # only the new samples, then update the 
                # cluster means using the learning rate
                # as the weight in a convex sum
                if self._mode == 'tangent_space':
                    sample_mean = np.mean(X[y == i], axis=0)
                    clust_mod.cluster_centers_[i] = (
                        (1-self._update_rate)*clust_mod.cluster_centers_[i] 
                        + self._update_rate*sample_mean
                    )
                else:
                    sample_mean = pyriemann.utils.mean.mean_covariance(X[y == i])
                    clust_mod.cluster_centers_[i] = pyriemann.utils.geodesic.geodesic_riemann(
                        clust_mod.cluster_centers_[i], sample_mean, alpha=self._update_rate
                    )


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

    def prune_state(self, state_idx):
        """
        Remove a state from the clustering model.

        Parameters
        ----------
        state_idx : int
            Index of the state to remove
        """
        if self._mode == 'potato':
            self._mod[-1].prune_state(state_idx)
        else:
            self._mod.named_steps['clustering'].cluster_centers_ = np.delete(
                self._mod.named_steps['clustering'].cluster_centers_, state_idx, axis=0
            )
        self._n_clusters -= 1

    def dump_mdl_to_file(self, file_path):
        """
        Save the clustering model to a file.

        Parameters
        ----------
        file_path : str
            Path to the file where the model will be saved.
        """
        import pickle
        # save the model to a file
        mdl = self._mod
        with open(file_path, 'wb') as f:
            pickle.dump(mdl, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def add_to_graph(
        cls, graph, inA, outA=None, outB=None,
        k_selection_method='pred_strength', 
        n_clusters=2, k_sel_thresh=0.5, 
        k_range=(2, 10), tangent_space=True,
        update_rate=0.1, outlier_state=False,
        scale_data=True, init_data=None, 
        init_labels=None
    ):
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
        outlier_state : bool, default=False
            Whether the model should include an outlier state. If True,
            the predictions will be made using a set of Riemannian Potato
            models. If the input data is not close to any of the potato
            models, it will be classified as an outlier.
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
        k = cls(
            graph, inA, outA, outB, k_selection_method,
            n_clusters, k_sel_thresh, k_range, tangent_space, scale_data,
            outlier_state, update_rate,
            init_data, init_labels
        )
        
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
            n.add_initialization_data([init_data], init_labels)

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


class MultiPotatoClustering(
    sklearn.base.BaseEstimator, 
    sklearn.base.ClassifierMixin,
    sklearn.base.ClusterMixin,
    sklearn.base.TransformerMixin
):
    
    """
    Defines an extension to the sklearn KMeans clustering algorithm
    that uses the KMeans clusters to define a set of K Riemannian 
    Potato models. The Potato models are used to classify the data
    into K+1 different classes. The (K+1)th class is the outlier class.


    """

    def __init__(self, n_clusters=2, threshold=2.5, space='tangent_space', scale=True):
        """Init."""
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.space = space
        self.scale = scale,
        self._state_models = []
        self._km = None


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
        self._state_models = []
        self._km = None

        # first use KMeans to define the initial set of clusters
        if self.space == 'tangent_space':
            km_pipe = []
            km_pipe.append(('tangent', pyriemann.tangentspace.TangentSpace()))
            if self.scale:
                km_pipe.append(('scaler', sklearn.preprocessing.StandardScaler()))
            km_pipe.append(('clustering', sklearn.cluster.KMeans(n_clusters=self.n_clusters, random_state=7164425)))
            self._km = sklearn.pipeline.Pipeline(km_pipe)
        else:
            self._km = pyriemann.clustering.Kmeans(n_clusters=self.n_clusters)

        print(f"Fitting KMeans with {self.n_clusters} clusters")
        self._km.fit(X)
        labels = self._km.predict(X)

        # create a set of Riemannian Potato models
        for i in range(self.n_clusters):
            # extract the data for the i-th cluster
            X_i = X[labels == i]

            # create a Potato model for the i-th cluster
            potato = pyriemann.clustering.Potato(threshold=self.threshold)
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
        zdists, rdists = self.transform(X)

        # check if any samples should be classified to
        # the outlier class
        rdists[zdists > self.threshold] = np.inf
        #min_dists = np.min(zdists, axis=1)
        y = np.argmin(rdists, axis=1)
        min_dists = np.array([zdists[i, y[i]] for i in range(X.shape[0])])
        y[min_dists > self.threshold] = self.n_clusters

        return y

    def predict_proba(self, X):
        """
        Predict the class probabilities of the input data.

        NOTE: The probability of the outlier class is not
        computed.
        """
        dists, _ = self.transform(X)
        # compute softmax based on negative distances
        probs = np.exp(-dists) / np.sum(np.exp(-dists), axis=1)[:, np.newaxis]
        
        # add zero probability for the outlier class
        probs = np.hstack((probs, np.zeros((probs.shape[0], 1))))

        return probs

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
        zdists = np.zeros((X.shape[0], self.n_clusters))
        rdists = np.zeros((X.shape[0], self.n_clusters))

        # compute the distance of the input data to each of the 
        # potato models
        for i, potato in enumerate(self._state_models):
            zdists[:,i] = potato.transform(X)
            for j, xj in enumerate(X):
                rdists[j,i] = pyriemann.utils.distance.distance_riemann(
                    xj, potato._mdm.covmeans_[0]
                )

        
        #print(f"Potato state distances: {dists}")
        return zdists, rdists
    

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

    def prune_state(self, state_idx):
        """
        Remove a state from the clustering model.

        Parameters
        ----------
        state_idx : int
            Index of the state to remove
        """
        self._state_models.pop(state_idx)
        self.n_clusters -= 1

